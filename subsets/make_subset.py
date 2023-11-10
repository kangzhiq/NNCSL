import sys
import os
from pathlib import Path
import argparse

from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet100'], help='Name of the datasets')
parser.add_argument('--seed', type=int, help='Random seed')
parser.add_argument('--percent', type=float, help='Percentage N for N% of labeled data')


def split(dataset_name, seed, percent):

    dataset_class = {
        'cifar10': CIFAR10,
        'cifar100': CIFAR100,
        'imagenet100': ImageFolder
    }[dataset_name]

    if dataset_name == 'imagenet100':
        dataset = dataset_class(Path('../datasets') / Path(dataset_name) / Path('train'))
    else:
        dataset = dataset_class(Path('../datasets') / Path(dataset_name))

    indices, _ = train_test_split(
        range(len(dataset)),
        train_size=percent/100,
        stratify=dataset.targets,
        random_state=seed)

    samples_per_class = len(indices) // len(dataset.classes)
    print(f'samples: {len(indices)}')
    print(f'samples per class: {samples_per_class}')

    subset_dir = Path(f'{dataset_name}')
    subset_dir.mkdir(parents=True, exist_ok=True)
    subset_file = subset_dir / f'{percent}%_seed{seed}.txt'
    print(f'writing to {subset_file}...')
    with open(subset_file, 'w') as f:
        for i in indices:
            f.write(f"{i}\n")

    # Add a file of classes
    subset_file = subset_dir / f'{percent}%_seed{seed}_cls.txt'
    print(f'writing to {subset_file}...')
    with open(subset_file, 'w') as f:
        for i in indices:
            f.write(f"{dataset.targets[i]}\n")


if __name__ == '__main__':
    args = parser.parse_args()
    split(dataset_name=args.dataset, seed=args.seed, percent=args.percent)