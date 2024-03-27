#!/bin/bash

echo -n "Experiment name: (1) super_resolution (2) diffusion -> "
read exp_name
if [ $exp_name -eq 1 ]; then
    echo "super_resolution"
    exp_name="super_resolution"
    echo -n "Model to use: (1) srcnn (2) srgan -> "
    read model_used
    if [ $model_used -eq 1 ]; then
        echo "srcnn"
        model_used="srcnn"
    elif [ $model_used -eq 2 ]; then
        echo "srgan"
        model_used="srgan"
    else
        echo "Invalid option"
        exit
    fi
elif [ $exp_name -eq 2 ]; then
    echo "diffusion"
    exp_name="diffusion"
    model_used="diff_unet"
else
    echo "Invalid option"
    exit
fi

if [ $model_used == "diff_unet" ]; then
    echo -n "No. of samples: "
    read num
    echo -n "Sampling technique: (1) DDPM (2) DDIM -> "
    read tech
    if [ $tech -eq 1 ]; then
        echo "DDPM"
        tech="ddpm"
    elif [ $tech -eq 2 ]; then
        echo "DDIM"
        tech="ddim"
    else
        echo "Invalid option"
        exit
    fi
    params="experiment.diffusion.diff_unet.test_param"
fi

if [ $exp_name == "diffusion" ]; then
    python3.10 eval.py experiment_name=$exp_name model_used=$model_used $params.samples=$num $params.sampling_technique=$tech

elif [ $exp_name == "super_resolution" ]; then
    python3.10 eval.py experiment_name=$exp_name model_used=$model_used
fi