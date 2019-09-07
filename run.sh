#!/bin/sh

# OPT == 1 -> VAD
# OPT == 2 -> resampling
# OPT == 3 -> VAD + resampling

OPT=3

python ./main.py --opt $OPT --path $PWD