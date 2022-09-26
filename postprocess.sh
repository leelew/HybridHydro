#!/bin/bash

saved_model_path="/tera08/lilu/HybridHydro/output/saved_model/"
saved_forecast_path="/tera08/lilu/HybridHydro/output/forecast/"
outputs_path="/tera08/lilu/HybridHydro/output/"
inputs_path="/tera08/lilu/HybridHydro/input/"
model_name="convlstm"

python3 -B src/postprocess.py --id $i --epochs $epochs --saved_model_path $saved_model_path --saved_forecast_path $saved_forecast_path --inputs_path $inputs_path --outputs_path $outputs_path --batch_size $batch_size --model_name $model_name

python3 -B src/cal_perf.py --id $i --epochs $epochs --saved_model_path $saved_model_path --saved_forecast_path $saved_forecast_path --inputs_path $inputs_path --outputs_path $outputs_path --batch_size $batch_size --model_name $model_name
