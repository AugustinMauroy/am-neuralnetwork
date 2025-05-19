#!/bin/sh

if [ "$(basename "$(pwd)")" != "example" ]; then
    cd "$(dirname "$0")"
fi

echo "Downloading CIFAR-10 and CIFAR-100 datasets..."
curl -O https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -xvf cifar-10-binary.tar.gz
rm cifar-10-binary.tar.gz
echo "CIFAR-10 dataset downloaded and extracted !"

echo "Downloading CIFAR-100 dataset..."
curl -O https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz
tar -xvf cifar-100-binary.tar.gz
rm cifar-100-binary.tar.gz
echo "CIFAR-100 dataset downloaded and extracted !"