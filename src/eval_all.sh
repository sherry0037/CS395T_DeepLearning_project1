#!/usr/bin/env bash

for i in 2 3 4 5
do
    python grade.py --DATASET_TYPE=yearbook --type=valid --model=$i.pb
done
