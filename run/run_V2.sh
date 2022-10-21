#!/bin/bash
#source /opt/miniconda3/bin/activate /opt/miniconda3/envs/lilu
root_path="/data/lilu"
work_path=$root_path"/HybridHydro/"
saved_model_path=$root_path"/HybridHydro/output/saved_model/"
saved_forecast_path=$root_path"/HybridHydro/output/forecast/"
outputs_path=$root_path"/HybridHydro/output/"
inputs_path=$root_path"/HybridHydro/input/"

nohup python3 -B $root_path/HybridHydro/V2/run.py --work_path $work_path --saved_model_path $saved_model_path --saved_forecast_path $saved_forecast_path --inputs_path $inputs_path --outputs_path $outputs_path >> $root_path/HybridHydro/logs/V2.log 2>&1 &
