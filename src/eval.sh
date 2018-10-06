#!/usr/bin/env bash

for i in 0 1 2 3 4 5
do
    python grade.py --DATASET_TYPE=yearbook --type=valid --model=trained_graph_%i.pb
done