#!/bin/bash

source /sonic_home/willianjunior/vllm-segment/prep.sh

python3 -u qrf-prep.py --prompts prompts.txt --random-forest-model models/random-forest-model-335.pkl.qrf

