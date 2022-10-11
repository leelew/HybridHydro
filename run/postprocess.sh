#!/bin/bash
source /home/lilu/anaconda3/bin/activate /home/lilu/anaconda3/envs/lilu
root_path="/work"
work_path=$root_path"/HybridHydro/"
saved_model_path=$root_path"/HybridHydro/output/saved_model/"
saved_forecast_path=$root_path"/HybridHydro/output/forecast/"
outputs_path=$root_path"/HybridHydro/output/"
inputs_path=$root_path"/HybridHydro/input/"
model_name="convlstm"

python3 -B src/postprocess.py --saved_model_path $saved_model_path --saved_forecast_path $saved_forecast_path --inputs_path $inputs_path --outputs_path $outputs_path --model_name $model_name

python3 -B src/cal_perf.py --saved_model_path $saved_model_path --saved_forecast_path $saved_forecast_path --inputs_path $inputs_path --outputs_path $outputs_path --model_name $model_name
