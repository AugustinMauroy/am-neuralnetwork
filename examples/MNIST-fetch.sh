#!/bin/bash

if [ "$(basename "$(pwd)")" != "example" ]; then
    cd "$(dirname "$0")"
fi

curl -L -o ./mnist-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset
unzip -o ./mnist-dataset.zip \
  -d ./mnist-dataset
rm -rf ./mnist-dataset.zip
