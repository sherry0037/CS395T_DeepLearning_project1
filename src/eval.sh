#!/usr/bin/env bash
%type=valid
type=test
%python grade.py --DATASET_TYPE=yearbook --type=$type --model=trained_graph_decades_1.pb --label_path=trained_labels_decades.txt --data_type=decade
python grade.py --DATASET_TYPE=yearbook --type=$type --model=1.pb --label_path=trained_labels.txt
