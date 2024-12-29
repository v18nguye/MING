#!/bin/bash 

export CUDA_VISIBLE_DEVICES=0
export WANDB__SERVICE_WAIT=300

while getopts d:e:t:n: flag
do
    case "${flag}" in
        d) dataset=${OPTARG};;
        e) experiment=${OPTARG};;
        t) task=${OPTARG};;
        n) name=${OPTARG};;
    esac
done

echo "run dataset: $dataset, task: $task, experiment: $experiment, name: $name";
if [ "$dataset" = "qm9" ]
    then 
        export SAMPLER_BATCH=5000
        export NUM_MOL_SAMPLER=10000
        export DATA_DIR=${PWD}'/data'
elif [ "$dataset" = "zinc" ]
    then
        export SAMPLER_BATCH=2024
        export NUM_MOL_SAMPLER=10000
        export DATA_DIR=${PWD}'/data'
elif [ "$dataset" = "moses" ]
    then 
        export SAMPLER_BATCH=2000
        export NUM_MOL_SAMPLER=25000
        export DATA_DIR=${PWD}'/processed_data/MOSES'
else
   echo "no dataset to run !!!"
fi


if [ "$task" = "diff" ]
    then
        python ./run_train.py data=$dataset +$experiment=$name
elif [ "$task" = "sample" ]
    then
        python ./run_sampling.py data=$dataset +$experiment=$name
else 
    echo "no task to run !!!"
fi