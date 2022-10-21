#!/bin/bash
root_path="/data/lilu"
work_path=$root_path"/HybridHydro/"
saved_model_path=$root_path"/HybridHydro/output/saved_model/"
saved_forecast_path=$root_path"/HybridHydro/output/forecast/"
outputs_path=$root_path"/HybridHydro/output/"
inputs_path=$root_path"/HybridHydro/input/"
num_jobs=25

for ((i=1; i<${num_jobs}; i++))
do
	nohup python3 -B $root_path/HybridHydro/V4/run.py --work_path $work_path --saved_model_path $saved_model_path --saved_forecast_path $saved_forecast_path --inputs_path $inputs_path --outputs_path $outputs_path --id $i >> $root_path/HybridHydro/logs/case-$i.log 2>&1 &
        sleep 20m
done
