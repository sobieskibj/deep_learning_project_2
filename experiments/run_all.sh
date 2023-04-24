#!/bin/bash
conda activate DL2
for script in experiments/*.py
do
    python $script
done