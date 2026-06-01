#!/bin/bash

for item in baseline pretrained scratch; do
    ./train_one.sh $item
done