#!/bin/bash

saved_model_path="/tera05/lilu/HybridHydro/output/saved_model/"
saved_forecast_path="/tera05/lilu/HybridHydro/output/forecast/"
outputs_path="/tera05/lilu/HybridHydro/output/"
inputs_path="/tera05/lilu/HybridHydro/input/"
model_name="convlstm_condition"

#python3 -B src/postprocess.py --saved_model_path $saved_model_path --saved_forecast_path $saved_forecast_path --inputs_path $inputs_path --outputs_path $outputs_path --model_name $model_name

python3 -B src/cal_perf.py --saved_model_path $saved_model_path --saved_forecast_path $saved_forecast_path --inputs_path $inputs_path --outputs_path $outputs_path --model_name $model_name
