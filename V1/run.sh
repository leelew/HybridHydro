#!/bin/bash
conda activate lilu
num_jobs=2
epochs=50

for ((i=1; i<${num_jobs}; i++))
do 
    python3 -B src/run.py --id $i --epochs $epochs
done
conda deactivate
