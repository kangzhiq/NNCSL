# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import time 
import logging
import sys
from collections import OrderedDict

import numpy as np
from tqdm import tqdm

from src.utils import (
    AllGather,
    AllReduce
)

import copy
import random

import torch
import torch.nn.functional as F
import src.resnet as resnet
from src.utils import (
    gpu_timer,
    init_distributed,
    WarmupCosineSchedule,
    CSVLogger,
    AverageMeter,
    make_buffer_lst
)
from src.losses import (
    init_paws_loss,
    make_labels_matrix
)
from src.data_manager import (
    init_data,
    make_transforms,
    make_multicrop_transform
)
from src.sgd import SGD
from src.lars import LARS

# import apex
from torch.nn.parallel import DistributedDataParallel

from src.utils import AllReduce

# --
log_timings = True
log_freq = 10
checkpoint_freq = 50
# --

_GLOBAL_SEED = 42
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
random.seed(_GLOBAL_SEED)

# # Deterministic mode
#####################
# # Uncomment the code below for fully deterministic runs
# # If activated and CUFA >= 10.2, please run the script with
# #              CUBLAS_WORKSPACE_CONFIG=:16:8 python --sel nncsl_train --fname xxxxxx
# #
# torch.backends.cudnn.benchmark = False
# torch.set_deterministic(True)

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.basicConfig(filename='output/cifar10_5%_buf500.log', level=logging.INFO)
logger = logging.getLogger()


def main(args):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #
    # -- META
    model_name = args['meta']['model_name']
    output_dim = args['meta']['output_dim']
    load_model = args['meta']['load_checkpoint']
    save_ckpt = args['meta']['save_checkpoint']
    start_task_idx = args['meta']['start_task_idx']
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    use_fp16 = args['meta']['use_fp16']
    w_paws = args['meta']['w_paws'] # weight for the paws loss
    w_me_max = args['meta']['w_me_max'] # weight for the me-max loss
    w_online = args['meta']['w_online'] # weight for the linear evaluation head
    w_dist = args['meta']['w_dist'] # weight for the distillation
    alpha = args['meta']['alpha'] # If 0, snn distillation, if 1, feature distillation
    use_pred_head = args['meta']['use_pred_head']
    device = torch.device(args['meta']['device'])
    torch.cuda.set_device(device)

    # -- CRITERTION
    reg = args['criterion']['me_max']
    supervised_views = args['criterion']['supervised_views']
    # classes_per_batch = args['criterion']['classes_per_batch']
    s_batch_size = args['criterion']['supervised_imgs_per_class']
    us_batch_size = args['criterion']['unlabeled_supervised_imgs_per_class']
    u_batch_size = args['criterion']['unsupervised_batch_size']
    temperature = args['criterion']['temperature']
    sharpen = args['criterion']['sharpen']

    # -- DATA
    unlabeled_frac = args['data']['unlabeled_frac']
    color_jitter = args['data']['color_jitter_strength']
    normalize = args['data']['normalize']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    dataset_name = args['data']['dataset']
    subset_path = args['data']['subset_path']
    subset_path_cls = args['data']['subset_path_cls']
    unique_classes = args['data']['unique_classes_per_rank']
    multicrop = args['data']['multicrop']
    us_multicrop = args['data']['us_multicrop']
    label_smoothing = args['data']['label_smoothing']
    data_seed = args['data']['data_seed']
    num_classes = {'cifar10': 10, 'cifar100': 100, 'imagenet': 100, 'tinyimagenet':10}[dataset_name]
    if 'cifar' in dataset_name or 'tiny' in dataset_name:
        data_seed = args['data']['data_seed']
        crop_scale = (0.75, 1.0) if multicrop > 0 else (0.5, 1.0)
        mc_scale = (0.3, 0.75)
        mc_size = 18
    else:
        crop_scale = (0.14, 1.0) if multicrop > 0 else (0.08, 1.0)
        mc_scale = (0.05, 0.14)
        mc_size = 96

    # -- OPTIMIZATION
    wd = float(args['optimization']['weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']
    lr_cls = args['optimization']['lr_cls']

    mom = args['optimization']['momentum']
    nesterov = args['optimization']['nesterov']

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    # wandb_log = args['logging']['wandb']

    # -- CONTINUAL 
    num_tasks = args['continual']['num_tasks']
    cl_setting = args['continual']['setting']
    mask = args['continual']['mask'] # Filtering
    detach = args['continual']['detach'] # if false: use Linear evaluation head
    buffer_size = args['continual']['buffer_size'] # Buffer size 
    # ----------------------------------------------------------------------- #

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_{num_tasks}T_r{rank}.csv')
    ckpt_name = f'{tag}_{num_tasks}T_p{w_paws}_m{w_me_max}_o{w_online}_d{w_dist}_a{alpha}_buffer{buffer_size}' + '-ep{epoch}-r4.pth.tar'
    save_path = os.path.join(folder, 'Task{task_id}', ckpt_name)
    latest_path = os.path.join(folder, f'{tag}_{num_tasks}T-latest.pth.tar')
    best_path = os.path.join(folder, f'{tag}_{num_tasks}T' + '-best.pth.tar')
    load_path = None
    if load_model:
        if num_epochs < 50:
            print('very small num_epochs {}, loaded epoch250 ckpt. Are you testing? Verify...........'.format(num_epochs))
            time.sleep(5)

        assert 0 < start_task_idx < num_tasks
        # Loading the ckpt from the previous task
        load_path = save_path.format(task_id=f'{start_task_idx-1}', epoch=f'{num_epochs}')

        if not os.path.isfile(load_path):
            raise ValueError('Checkpoint path does not exist: {}'.format(load_path))
            

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'paws-xent-loss'),
                           ('%.5f', 'paws-me_max-reg'),
                           ('%d', 'time (ms)'))

    # -- init model
    encoder = init_model(
        device=device,
        model_name=model_name,
        use_pred=use_pred_head,
        output_dim=output_dim,
        cifar='cifar' in args['data']['dataset'],
        num_classes=num_classes,
        detach=detach)
    

    # -- init losses
    paws, snn, sharpen = init_paws_loss(
        multicrop=multicrop,
        tau=temperature,
        T=sharpen,
        me_max=reg)


    # -- make data transforms
    transform, init_transform = make_transforms(
        dataset_name=dataset_name,
        subset_path=subset_path,
        unlabeled_frac=unlabeled_frac,
        training=True,
        split_seed=data_seed,
        crop_scale=crop_scale,
        basic_augmentations=False,
        color_jitter=color_jitter,
        normalize=normalize)
    multicrop_transform = (multicrop, None)
    if multicrop > 0:
        multicrop_transform = make_multicrop_transform(
                dataset_name=dataset_name,
                num_crops=multicrop,
                size=mc_size,
                crop_scale=mc_scale,
                normalize=normalize,
                color_distortion=color_jitter)

    if us_multicrop > 0:
        us_multicrop_transform = make_multicrop_transform(
                dataset_name=dataset_name,
                num_crops=us_multicrop,
                size=mc_size,
                crop_scale=mc_scale,
                normalize=normalize,
                color_distortion=color_jitter)

    # -- make validation data transforms
    val_dataset_name = dataset_name + '_fine_tune'
    val_transform, val_init_transform = make_transforms(
        dataset_name=val_dataset_name,
        subset_path=subset_path,
        unlabeled_frac=-1,
        training=False,
        basic_augmentations=False,
        force_center_crop=False,
        normalize=normalize)

    # -- init optimizer and scheduler
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    encoder, optimizer = init_opt(
        encoder=encoder,
        weight_decay=wd,
        ref_lr=lr,
        nesterov=nesterov,
        ref_lr_cls=lr_cls)

    if world_size > 1:
        encoder = DistributedDataParallel(encoder, broadcast_buffers=False)

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        encoder, optimizer, start_epoch = load_checkpoint(
            r_path=load_path,
            encoder=encoder,
            opt=optimizer,
            scaler=scaler,
            use_fp16=use_fp16)
        print('model loaded from {}'.format(load_path))
        time.sleep(2)

    classes = list(range(num_classes))
    classes_per_task = num_classes // num_tasks
    tasks = [classes[i:i + classes_per_task] for i in range(0, len(classes), classes_per_task)]

    # if not empty, the buffer should contain data for at least one batch
    # otherwise, reduce the batch size for labeled data
    if buffer_size > 0:
        assert buffer_size >= s_batch_size * len(sum(tasks[:num_tasks-1], []))

    # -- continual training loop
    pre_encoder = None
    filtered_proportion = 0.2
    filtered_channels = int(output_dim * filtered_proportion)
    buffer_lst = None
    num_in_buffer = 0

    # make the log for wandb and  buffer_lst
    if load_model and start_task_idx != 0 :
        print('Skipping the first {} tasks...'.format(start_task_idx))
        time.sleep(3)
        for tt in range(start_task_idx):
            for epoch in range(0, num_epochs):
                log_dict = {
                    'paws_loss': 0,
                    'me_max_loss': 0,
                    'online_eval_loss': 0,
                    'dist_loss': 0,
                    'logit_dist_loss': 0,
                    'train_acc1_lab': 0,
                    'train_acc5_lab': 0,
                    'train_acc1_unlab': 0,
                    'train_acc5_unlab': 0,
                    'step_time': 0,
                    'data_time': 0,
                    'epoch': epoch,
                    'lr': 0,
                    'lr_cls': 0
                }
            buffer_lst = make_buffer_lst(buffer_lst, buffer_size, subset_path, subset_path_cls, tasks, tt)
        start_epoch = 0
        pre_encoder = copy.deepcopy(encoder.eval())
    
    if (not load_model) and start_task_idx != 0 :
        raise ValueError('Starting at Task {} but not loading?? What do you want?'.format(start_task_idx))

    for task_idx in range(start_task_idx, num_tasks):

        if cl_setting == 'seen_current':
            visible_class_ul = sum(tasks[:task_idx + 1], [])
        elif cl_setting == 'current':
            visible_class_ul = tasks[task_idx]
        elif cl_setting == 'all':
            visible_class_ul = sum(tasks, [])
        else:
            raise ValueError('unknown setting!')
        # -- assume support images are sampled with ClassStratifiedSampler
        num_cur_classes = len(tasks[task_idx])
        pre_classes = sum(tasks[:task_idx], [])
        num_pre_classes = len(pre_classes)
        seen_classes = sum(tasks[:task_idx + 1], [])
        num_seen_classes = len(seen_classes)
        if buffer_size == 0:
            num_classes_cl = num_cur_classes
        else:
            if mask:
                num_classes_cl = num_cur_classes
            else:
                num_classes_cl = num_seen_classes

        labels_matrix = make_labels_matrix(
            num_classes=num_classes_cl,
            s_batch_size=s_batch_size,
            world_size=world_size,
            device=device,
            unique_classes=unique_classes,
            smoothing=label_smoothing,
            task_idx=task_idx)
        
        if pre_encoder is not None:
            if buffer_size == 0:
                pre_labels_matrix = make_labels_matrix(
                    num_classes=num_cur_classes,
                    s_batch_size=s_batch_size,
                    world_size=world_size,
                    device=device,
                    unique_classes=unique_classes,
                    smoothing=label_smoothing,
                    task_idx=task_idx)
            else:
                pre_labels_matrix = make_labels_matrix(
                    num_classes=num_pre_classes,
                    s_batch_size=s_batch_size,
                    world_size=world_size,
                    device=device,
                    unique_classes=unique_classes,
                    smoothing=label_smoothing,
                    task_idx=task_idx)
        # print('-----------------label matrix; {}-------------'.format(labels_matrix))
        # -- init data-loaders/samplers
        if buffer_size == 0:
            classes_per_batch = num_cur_classes
        else:
            classes_per_batch = num_seen_classes
        (unsupervised_loader,
        unsupervised_sampler,
        supervised_loader,
        supervised_sampler) = init_data(
            dataset_name=dataset_name,
            transform=transform,
            init_transform=init_transform,
            supervised_views=supervised_views,
            u_batch_size=u_batch_size,
            s_batch_size=s_batch_size,
            unique_classes=unique_classes,
            classes_per_batch=classes_per_batch,
            multicrop_transform=multicrop_transform,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            training=True,
            copy_data=copy_data,
            tasks=tasks,
            task_idx=task_idx,
            visible_class_ul=visible_class_ul,
            buffer_lst=buffer_lst)
        iter_supervised = None

        ipe = len(unsupervised_loader)
        logger.info(f'iterations per epoch: {ipe}')
        logger.info(f'initialized supervised data-loader (ipe {len(supervised_loader)})')

        # -- make val data transforms and data loaders/samples
        val_loader, _ = init_data(
            dataset_name=val_dataset_name,
            transform=val_transform,
            init_transform=val_init_transform,
            u_batch_size=None,
            s_batch_size=100,
            classes_per_batch=None,
            world_size=1,
            rank=0,
            root_path=root_path,
            image_folder=image_folder,
            training=False,
            copy_data=copy_data,
            tasks=tasks,
            task_idx=task_idx,
            visible_class_ul=seen_classes)
        logger.info(f'initialized val data-loader (ipe {len(val_loader)})')

        # setting1
        # temp_lr_lst = [1.2, 1.1, 1, 0.9 ,0.8]
        # temp_lr = temp_lr_lst[task_idx]
        scheduler = init_scheduler(
            optimizer=optimizer,
            start_lr=start_lr,
            ref_lr=lr,
            final_lr=final_lr,
            iterations_per_epoch=ipe,
            warmup=warmup,
            num_epochs=num_epochs)

        if task_idx == start_task_idx and load_model:
            for _ in range(start_epoch):
                for _ in range(ipe):
                    scheduler.step()

        # -- TRAINING LOOP
        best_loss = None
        for epoch in range(start_epoch, num_epochs):
            logger.info('Epoch %d' % (epoch + 1))

            # if epoch % 50 == 0:
            #     encoder.reset_parameters()
            #     # encoder.partial_reset_parameters(reset_prob=0.7)
            #     print('-------------Reset params...--------------')

            # -- update distributed-data-loader epoch
            unsupervised_sampler.set_epoch(epoch)
            if supervised_sampler is not None:
                supervised_sampler.set_epoch(epoch)

            loss_meter = AverageMeter()
            ploss_meter = AverageMeter()
            rloss_meter = AverageMeter()
            online_eval_meter = AverageMeter()
            dist_meter = AverageMeter()
            logit_dist_meter = AverageMeter()
            sacc1_meter = AverageMeter()
            sacc5_meter = AverageMeter()
            uacc1_meter = AverageMeter()
            uacc5_meter = AverageMeter()
            time_meter = AverageMeter()
            data_meter = AverageMeter()

            # a set pf averagemeters for labeled data
            meter_set_l = {}
            for tt_idx in range(task_idx+1):
                meter_set_l[tt_idx] = AverageMeter()
            # a set pf averagemeters for unlabeled data
            meter_set_u = {}
            for tt_idx in range(task_idx+1):
                meter_set_u[tt_idx] = AverageMeter()

            cur_lr = optimizer.param_groups[0]['lr']
            cur_lr_cls = optimizer.param_groups[-1]['lr']

            for itr, udata in enumerate(unsupervised_loader):
                def make_one_hot_label(slabels, num_class):
                    total_images = slabels.shape[0]
                    labels = torch.zeros(total_images, num_class).to(device)
                    for i, label in enumerate(slabels):
                        labels[i, label] = 1
                    return labels
                
                def cross_entropy_with_logits(logits, targets):
                    return  -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * targets, dim=1))

                def standardize(x):
                    return (x - x.mean()) / x.std()

                def load_imgs():

                    # -- unsupervised imgs
                    uimgs = [u.to(device, non_blocking=True) for u in udata[:-1]]
                    ulabels = udata[-1].to(device, non_blocking=True).repeat(2)

                    # -- supervised imgs
                    global iter_supervised
                    try:
                        sdata = next(iter_supervised)
                    except Exception:
                        iter_supervised = iter(supervised_loader)
                        logger.info(f'len.supervised_loader: {len(iter_supervised)}')
                        sdata = next(iter_supervised)
                    finally:
                        slabels = sdata[-1].to(device, non_blocking=True).repeat(2)
                        omatrix = make_one_hot_label(sdata[-1].to(device, non_blocking=True), num_seen_classes)
                        plabels = torch.cat([labels_matrix for _ in range(supervised_views)])   
                        olabels = torch.cat([omatrix for _ in range(supervised_views)])                        
                        simgs = [s.to(device, non_blocking=True) for s in sdata[:-1]]

                    # -- concatenate supervised imgs and unsupervised imgs
                    imgs = simgs + uimgs
                    return imgs, plabels, slabels, ulabels, olabels
                (imgs, plabels, slabels, ulabels, olabels), dtime = gpu_timer(load_imgs)
                data_meter.update(dtime)

                def train_step():
                    with torch.cuda.amp.autocast(enabled=use_fp16):
                        optimizer.zero_grad()

                        # --
                        # h: representations of 'imgs' before head
                        # z: representations of 'imgs' after head
                        # -- If use_pred_head=False, then encoder.pred (prediction
                        #    head) is None, and _forward_head just returns the
                        #    identity, z=h
                        h, z, l = encoder(imgs, return_before_head=True)
                        if pre_encoder is not None:
                            with torch.no_grad():
                                pre_h, pre_z, pre_l = pre_encoder(imgs, return_before_head=True)
                            h_proj = encoder.feat_proj(h)

                        # Compute paws loss in full precision
                        with torch.cuda.amp.autocast(enabled=False):

                            # Step 1. convert representations to fp32
                            h, z, l = h.float(), z.float(), l.float()
                            if pre_encoder is not None:
                                pre_h, pre_z, pre_l = pre_h.float(), pre_z.float(), pre_l.float() 
                                h_proj = h_proj.float()
                            

                            # Step 2. determine anchor views/supports and their
                            #         corresponding target views/supports
                            # --
                            if buffer_size == 0:
                                num_support_mix = (supervised_views) * s_batch_size * num_cur_classes
                            else:
                                num_support_mix = (supervised_views) * s_batch_size * num_seen_classes
                            num_u_data_mix =  u_batch_size 
                            # --
                            if mask:
                                labels_mask = [label in tasks[task_idx] for label in slabels]
                            else:
                                labels_mask = [label in seen_classes for label in slabels]

                            plabels_masked= plabels
                            anchor_supports = z[:num_support_mix][labels_mask]
                            anchor_views = z[num_support_mix:]
                            # --
                            target_supports = h[:num_support_mix].detach()[labels_mask]
                            target_views = h[num_support_mix:].detach()
                            target_views = torch.cat([
                                target_views[num_u_data_mix:2*num_u_data_mix],
                                target_views[:num_u_data_mix]], dim=0)

                            # Step 3. compute paws loss with me-max regularization
                            (ploss, me_max, probs_anchor) = paws(
                                anchor_views=anchor_views,
                                anchor_supports=anchor_supports,
                                anchor_support_labels=plabels_masked,
                                target_views=target_views,  
                                target_supports=target_supports,
                                target_support_labels=plabels_masked,
                                mask=labels_mask)

                            # Step 4. compute online eval loss
                            slogits = l[:num_support_mix,:num_seen_classes]
                            # # # Change the targets to onehot label for mix up
                            # online_eval_loss = cross_entropy_with_logits(slogits, olabels)
                            
                            online_eval_loss = F.cross_entropy(slogits, slabels)


                            # Step 5. Distillation
                            dist_loss = torch.tensor(0)
                            dist_logit_loss = torch.tensor(0)
                            if pre_encoder is not None:
                                mse_loss = torch.nn.MSELoss()     
                                sigmoid = torch.nn.Sigmoid() 
                                softmax = torch.nn.Softmax(dim=1)  
                                cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                                if buffer_size == 0:
                                    pre_mask = [True for label in slabels]
                                else:
                                    pre_mask = [label in pre_classes for label in slabels]
                                
                                # Distillation on features
                                dist_loss = mse_loss(h_proj, pre_h)

                                # Distillation on pseudo labels
                                # plabels for distillation with pre labeled data
                                pre_plabels = torch.cat([pre_labels_matrix for _ in range(supervised_views)])
                                cur_anchor_supports = z[:num_support_mix][pre_mask]
                                cur_anchor_views = z[num_support_mix:]
                                pre_anchor_supports = pre_z[:num_support_mix][pre_mask]
                                pre_anchor_views = pre_z[num_support_mix:]
                                pre_target_views = pre_z[num_support_mix:].detach()
                                pre_target_views = torch.cat([pre_target_views[num_u_data_mix:2*num_u_data_mix],
                                                               pre_target_views[:num_u_data_mix]], dim=0)

                                # Distillation with anchor views (snn)
                                cur_probs = snn(cur_anchor_views, cur_anchor_supports,  pre_plabels)
                                pre_probs = snn(pre_anchor_views, pre_anchor_supports,  pre_plabels)
                                dist_logit_loss = torch.mean(torch.sum(torch.log(cur_probs**(-pre_probs)), dim=1))
                                

                            loss = w_paws*ploss + w_me_max*me_max  + w_online*online_eval_loss + w_dist*(alpha*dist_loss + (1-alpha)*dist_logit_loss)

                    scaler.scale(loss).backward()
                    lr_stats = scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                    # calculate training accuracy on the supervised set
                    top1_correct = slogits.max(dim=-1).indices.eq(slabels).sum()
                    if num_tasks > 0:
                        val_topk = min(int(num_classes / num_tasks), 5)
                    else:
                        val_topk = 5
                    top5_correct = slogits.topk(val_topk, dim=-1).indices.eq(slabels.unsqueeze(1)).sum()

                    sacc1 = 100. * AllReduce.apply(top1_correct).item() / slogits.size(0)
                    sacc5 = 100. * AllReduce.apply(top5_correct).item() / slogits.size(0)

                    # calculate training accuracy on the unsupervised set
                    ulogits = l[num_support_mix:num_support_mix + 2 * num_u_data_mix]
                    top1_correct = ulogits.max(dim=-1).indices.eq(ulabels).sum()
                    if num_tasks > 0:
                        val_topk = min(int(num_classes / num_tasks), 5)
                    else:
                        val_topk = 5
                    top5_correct = ulogits.topk(val_topk, dim=-1).indices.eq(ulabels.unsqueeze(1)).sum()
                    uacc1 = 100. * AllReduce.apply(top1_correct).item() / ulogits.size(0)
                    uacc5 = 100. * AllReduce.apply(top5_correct).item() / ulogits.size(0)
                    # Adding per task training accuracy for supervised set:

                    return (
                        AllReduce.apply(loss).item(),
                        AllReduce.apply(ploss).item(),
                        AllReduce.apply(me_max).item(),
                        AllReduce.apply(online_eval_loss).item(),
                        AllReduce.apply(dist_loss).item(),
                        AllReduce.apply(dist_logit_loss).item(),
                        sacc1,
                        sacc5,
                        uacc1,
                        uacc5,
                        lr_stats)

                (loss, ploss, rloss, online_eval_loss, dist_loss, dist_logit_loss,
                sacc1, sacc5, uacc1, uacc5, lr_stats), etime = gpu_timer(train_step)
                loss_meter.update(loss)
                ploss_meter.update(ploss)
                rloss_meter.update(rloss)
                online_eval_meter.update(online_eval_loss)
                dist_meter.update(dist_loss)
                logit_dist_meter.update(dist_logit_loss)
                sacc1_meter.update(sacc1)
                sacc5_meter.update(sacc5)
                uacc1_meter.update(uacc1)
                uacc5_meter.update(uacc5)
                time_meter.update(etime)

                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    csv_logger.log(epoch + 1, itr,
                                ploss_meter.avg,
                                rloss_meter.avg,
                                time_meter.avg)
                    logger.info('[%d, %5d] loss: %.3f (%.3f %.3f %.3f %.3f %.3f) '
                                'acc1/5: (s: %.3f %.3f, u: %.3f %.3f) '
                                'time: (%dms %dms)'
                                % (epoch + 1, itr,
                                loss_meter.avg,
                                ploss_meter.avg,
                                rloss_meter.avg,
                                online_eval_meter.avg,
                                dist_meter.avg,
                                logit_dist_meter.avg,
                                sacc1_meter.avg,
                                sacc5_meter.avg,
                                uacc1_meter.avg,
                                uacc5_meter.avg,
                                time_meter.avg,
                                data_meter.avg))
                    if lr_stats is not None:
                        logger.info('[%d, %5d] lr_stats: %.3f (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                    lr_stats.avg,
                                    lr_stats.min,
                                    lr_stats.max))

                if np.isnan(loss):
                    save_dict = {
                        'encoder': encoder.state_dict(),
                        'preencoder': pre_encoder.state_dict(),
                        'imgs': imgs.cpu(),
                        'plabels': plabels,
                        'slabels': slabels,
                        'ulabels': ulabels,
                        'opt': optimizer.state_dict(),
                        'epoch': epoch + 1,
                        'unlabel_prob': unlabeled_frac,
                        'loss': loss_meter.avg,
                        's_batch_size': s_batch_size,
                        'u_batch_size': u_batch_size,
                        'world_size': world_size,
                        'lr': lr,
                        'temperature': temperature,
                        'amp': scaler.state_dict()
                    }
                    print('loss is nan, saving data....')
                    torch.save(save_dict, './loss_nan.pth.tar')

                assert not np.isnan(loss), 'loss is nan'

            # -- logging/checkpointing
            logger.info('avg. loss %.3f' % loss_meter.avg)

            if rank == 0:
                save_dict = {
                    'encoder': encoder.state_dict(),
                    'opt': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'unlabel_prob': unlabeled_frac,
                    'loss': loss_meter.avg,
                    's_batch_size': s_batch_size,
                    'u_batch_size': u_batch_size,
                    'world_size': world_size,
                    'lr': lr,
                    'temperature': temperature,
                    'amp': scaler.state_dict()
                }

                # Only save the ckpt for the last epoch
                # if num_epochs >= 100 and ((epoch + 1) % num_epochs == 0:
                if save_ckpt and (epoch + 1) % num_epochs == 0:
                    if num_epochs < 100:
                        print('very small epoch {}, do you really want to store this checkpoint??'.format(num_epochs))
                        time.sleep(5)
                    # Check dir:
                    ckpt_path = save_path.format(task_id=f'{task_idx}', epoch=f'{epoch + 1}')
                    dir_path = os.path.join(*ckpt_path.split('/')[:-1])
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    print('saving the checkpoint to {}'.format(ckpt_path))
                    torch.save(save_dict, ckpt_path)

                log_dict = {
                    'paws_loss': ploss_meter.avg,
                    'me_max_loss': rloss_meter.avg,
                    'online_eval_loss': online_eval_meter.avg,
                    'dist_loss': dist_meter.avg,
                    'logit_dist_loss': logit_dist_meter.avg,
                    'train_acc1_lab': sacc1_meter.avg,
                    'train_acc5_lab': sacc5_meter.avg,
                    'train_acc1_unlab': uacc1_meter.avg,
                    'train_acc5_unlab': uacc5_meter.avg,
                    'step_time': time_meter.avg,
                    'data_time': data_meter.avg,
                    'epoch': epoch,
                    'lr': cur_lr,
                    'lr_cls': cur_lr_cls
                }

                val_freq = 1
                if (epoch + 1) % val_freq == 0:
                    val_acc1, val_acc5, per_task_info = validate(val_loader, encoder, device, num_seen_classes, task_idx, tasks)
                    log_dict.update(**{'val_acc1': val_acc1, 'val_acc5': val_acc5})
                    logger.info(f'val acc1/5:  ({val_acc1} {val_acc5})')

                    for k, v in per_task_info.items():
                        log_dict.update(**{'val_acc1_task_{}'.format(k): v})
                    for k, v in meter_set_l.items():
                        log_dict.update(**{'train_labeled_acc1_task_{}'.format(k): v.avg})


                   
        # Save a copy of the curernt model for distillation
        pre_encoder = copy.deepcopy(encoder.eval())

        # Update the buffer
        buffer_lst = make_buffer_lst(buffer_lst, buffer_size, subset_path, subset_path_cls, tasks, task_idx)


@torch.no_grad()
def validate(val_loader, encoder, device, num_seen_classes, task_id=None, tasks=None):
    preds, labels = [], []
    for img, label in tqdm(val_loader, desc='validating'):
        img = img.to(device)
        _, logits = encoder._forward_backbone(img)
        logits = logits[:,:num_seen_classes]
        if task_id is not None:
            val_topk = min(int(num_seen_classes / (task_id+1)), 5)
        else:
            val_topk = 5
        preds.append(logits.cpu().topk(val_topk, dim=1).indices)
        labels.append(label)

    preds = torch.cat(preds)
    labels = torch.cat(labels)
    # Adding per task acc
    per_task_info = {}
    count = 0
    for t_id in range(task_id+1):
        cls_lst = tasks[t_id]
        idx_lst = torch.tensor([label in cls_lst for label in labels])
        top1_task = float(preds[:, 0][idx_lst].eq(labels[idx_lst]).sum()) 
        acc1 = 100. * top1_task / idx_lst.sum()
        count += idx_lst.sum()
        per_task_info[t_id] = acc1
    assert count == labels.size(0)
    top1_correct = float(preds[:, 0].eq(labels).sum())
    top5_correct = float(preds.eq(labels.unsqueeze(1)).sum())
    acc1 = 100. * top1_correct / labels.size(0)
    acc5 = 100. * top5_correct / labels.size(0)
    return acc1, acc5, per_task_info


def load_checkpoint(
    r_path,
    encoder,
    opt,
    scaler,
    use_fp16=False
):
    checkpoint = torch.load(r_path, map_location='cpu')
    epoch = checkpoint['epoch']

    # -- loading encoder
    encoder.load_state_dict(checkpoint['encoder'])
    logger.info(f'loaded encoder from epoch {epoch}')

    # -- loading optimizer
    opt.load_state_dict(checkpoint['opt'])
    if use_fp16:
        scaler.load_state_dict(checkpoint['amp'])
    logger.info(f'loaded optimizers from epoch {epoch}')
    logger.info(f'read-path: {r_path}')
    del checkpoint
    return encoder, opt, epoch


def init_model(
    device,
    model_name='resnet50',
    use_pred=False,
    output_dim=128,
    cifar=False,
    num_classes=100,
    detach=None
):
    assert detach is not None
    if 'wide_resnet' in model_name:
        encoder = wide_resnet.__dict__[model_name](dropout_rate=0.0)
        hidden_dim = 128
    else:
        encoder = resnet.__dict__[model_name](cifar=cifar, num_classes=num_classes, detach=detach)
        if model_name == 'resnet18':
            hidden_dim = 512
        else:
            hidden_dim = 2048
            if 'w2' in model_name:
                hidden_dim *= 2
            elif 'w4' in model_name:
                hidden_dim *= 4

    # -- projection head
    encoder.fc = torch.nn.Sequential(OrderedDict([
        ('fc1', torch.nn.Linear(hidden_dim, hidden_dim)),
        ('bn1', torch.nn.BatchNorm1d(hidden_dim)),
        ('relu1', torch.nn.ReLU(inplace=True)),
        ('fc2', torch.nn.Linear(hidden_dim, hidden_dim)),
        ('bn2', torch.nn.BatchNorm1d(hidden_dim)),
        ('relu2', torch.nn.ReLU(inplace=True)),
        ('fc3', torch.nn.Linear(hidden_dim, output_dim))
    ]))

    # -- projection head for feature alignment
    encoder.feat_proj = torch.nn.Sequential(OrderedDict([
        ('fc1', torch.nn.Linear(output_dim, output_dim)),
    ]))


    # -- prediction head
    encoder.pred = None
    if use_pred:
        mx = 4  # 4x bottleneck prediction head
        pred_head = OrderedDict([])
        pred_head['bn1'] = torch.nn.BatchNorm1d(output_dim)
        pred_head['fc1'] = torch.nn.Linear(output_dim, output_dim//mx)
        pred_head['bn2'] = torch.nn.BatchNorm1d(output_dim//mx)
        pred_head['relu'] = torch.nn.ReLU(inplace=True)
        pred_head['fc2'] = torch.nn.Linear(output_dim//mx, output_dim)
        encoder.pred = torch.nn.Sequential(pred_head)

    encoder.to(device)
    logger.info(encoder)
    return encoder

def init_scheduler(
    optimizer,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    final_lr=0.0,
):
    return WarmupCosineSchedule(
        optimizer,
        warmup_steps=warmup*iterations_per_epoch,
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=num_epochs*iterations_per_epoch)

def init_opt(
    encoder,
    ref_lr,
    nesterov,
    weight_decay=1e-6,
    ref_lr_cls=0.25
):
    param_groups = [
        {'params': (p for n, p in encoder.named_parameters()
                    if ('bias' not in n) and ('bn' not in n) and ('classifier' not in n))},
        {'params': (p for n, p in encoder.named_parameters()
                    if (('bias' in n) or ('bn' in n)) and ('classifier' not in n)),
         'LARS_exclude': True,
         'weight_decay': 0},
        {'params': (p for n, p in encoder.named_parameters() if ('classifier' in n)),
         'weight_decay': 0, 'lr': ref_lr_cls}
    ]
    optimizer = SGD(
        param_groups,
        weight_decay=weight_decay,
        momentum=0.9,
        nesterov=nesterov,
        lr=ref_lr)
    
    optimizer = LARS(optimizer, trust_coefficient=0.001)
    return encoder, optimizer


if __name__ == "__main__":
    main()
