#!/bin/bash

num_jobs=13
epochs=40
saved_model_path="/tera08/lilu/HybridHydro/output/saved_model/"
saved_forecast_path="/tera08/lilu/HybridHydro/output/forecast/"
outputs_path="/tera08/lilu/HybridHydro/output/"
inputs_path="/tera08/lilu/HybridHydro/input/"
model_name="convlstm"
batch_size=32

for ((i=7; i<${num_jobs}; i++))
do 
	nohup python3 -B src/run.py --id $i --epochs $epochs --saved_model_path $saved_model_path --saved_forecast_path $saved_forecast_path --inputs_path $inputs_path --outputs_path $outputs_path --batch_size $batch_size --model_name $model_name >> logs/case-$i.log 2>&1 &
done
