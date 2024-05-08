<div align="center">
  
  <div>
  <h1>A soft nearest-neighbor framework for continual semi-supervised learning</h1>
  </div>

  <div>
      Zhiqi Kang*&emsp; Enrico Fini*&emsp; Moin Nabi&emsp; Elisa Ricci&emsp; Karteek Alahari
  </div>

  <div>
      <h4>
          Oral presentation
      </h4>
  </div>
  <br/>

</div>

Official PyTorch implementation of our ICCV 2023 paper "[A soft nearest-neighbor framework for continual semi-supervised learning](https://arxiv.org/abs/2212.05102)". 

Our proposed NNCSL（**N**earest-**N**eighbor for **C**ontinual **S**emi-supervised **L**earning) is composed of a continual semi-supervised learner **CSL** and a distillation loss **NND**.

### CSL
is our base continual semi-supervised learner, which is developed from PAWS and adapted to a novel continual semi-supervised scenario. An
illustration of the architecture is as follows:

<p align="center">
    <img width="600" src="https://github.com/kangzhiq/NNCSL/blob/main/Imgs/method.png" alt="CSL">
</p>

### NND
is our propsed distillation strategy based on the nearest-neighbor classifier that transfers both class-level and feature-level knowledge. An
illustration of the architecture is as follows:

<p align="center">
    <img width="600" src="https://github.com/kangzhiq/NNCSL/blob/main/Imgs/NND.png" alt="NND">
</p>


## How to run NNCSL?

### Build the conda environment

Our implementation does not requires many special packages, but please make sure that the following requirements are satisfied in your environment:

- Python 3.8
- PyTorch install 1.7.1
- torchvision
- CUDA 11.0
- Apex with CUDA extension
- Other dependencies: PyYaml, numpy, opencv, submitit

### Download dataset
For CIFAR-10 and CIFAR-100, the datasets can be auto-downloaded by torchvision. 
For ImageNet-100, please download the dataset and make sure that the images are organised as follows:

<pre>
--configs  
--src  
--datasets  
  |--imagenet100  
      |--train  
           |--n01330764  
           |--...  
      |--val  
           |--n01330764  
           |--...  
</pre>

It is also possible to change the organization of the images and the path to the datasets. Please refer to the config files for more details

### Run the scripts
Once the dataset is ready, the experiment can be launched by the following commands:

#### CIFAR-10
for CIFAR-10, 0.8% of labeled data, buffer size 500, using our NNCSL

    python main.py --sel nncsl_train  --fname configs/nncsl/cifar10/cifar10_0.8%_buffer500_nncsl.yaml

for CIFAR-10, 0.8% of labeled data, buffer size 500, using CSL

    python main.py --sel nncsl_train  --fname configs/nncsl/cifar10/cifar10_0.8%_buffer500_csl.yaml

for CIFAR-10, 0.8% of labeled data, buffer size 500, using PAWS

    python main.py --sel nncsl_train  --fname configs/nncsl/cifar10/cifar10_0.8%_buffer500_paws.yaml

#### CIFAR-100
for CIFAR-100, 0.8% of labeled data, buffer size 500, using our NNCSL

    python main.py --sel nncsl_train  --fname configs/nncsl/cifar100/cifar100_0.8%_buffer500_nncsl.yaml

#### ImageNet-100
for ImageNet-100, 1% of labeled data, buffer size 500, using our NNCSL

    python main.py --sel nncsl_train  --fname configs/nncsl/imagenet100/imgnt100_1%_buffer500_nncsl.yaml

#### Change buffer size
one can easily change the buffer size by modifying the parameter `buffer_size` in the config file, we provide one example as:
for CIFAR-100, 0.8% of labeled data, buffer size 5120, using our NNCSL

    python main.py --sel nncsl_train  --fname configs/nncsl/cifar100/cifar100_0.8%_buffer5120_nncsl.yaml

Please check the config files in ./configs/nncsl/ for more different settings.

#### Change labeled data proportion
one can try with different ratio of labeled data other than the given settings (0.8%, 1%, 5% and 25%). There are two steps:
1. Generate the index files with `make_subset.py` in \subsets.
Below is an example for generate index files for CIFAR-10 with 50 percents of labeled data.

        python make_subset.py --dataset cifar10 --seed 0 --percent 50

2. Source the index files in the corresponding config files
Change `subset_path` and `subset_path_cls` to the corresponding path. Please note that these two parameters should be consistent.

### Reproduction
The training log files are shared in the /log directory for debugging reproduction issues. Please don't hesitate to contact us if you find any further difficulties.  

### Work in progress
We will keep updating our repository to make it easier to use and to share our recent progress on this project.
1. Using standard reservoir replay buffer

## Acknolegment
This implementation is developed based on the source code of [PAWS](https://github.com/facebookresearch/suncet).

## CITATION
If you find our codes or paper useful, please consider giving us a star or cite with:
```
@InProceedings{Kang_2023_ICCV,
    author    = {Kang, Zhiqi and Fini, Enrico and Nabi, Moin and Ricci, Elisa and Alahari, Karteek},
    title     = {A Soft Nearest-Neighbor Framework for Continual Semi-Supervised Learning},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {11868-11877}
}
```
