#!/bin/bash
  
num_jobs=2
epochs=40

for ((i=1; i<${num_jobs}; i++))
do
    nohup python3 -B src/run.py --id $i --epochs $epochs >> logs/case-$i.log 2>&1 &
done
