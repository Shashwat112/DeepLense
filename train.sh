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
        echo -n "Pre-train: (1) True (2) False -> "
        read pretrain
        if [ $pretrain -eq 1 ]; then
            echo "True"
            pretrain="True"
        elif [ $pretrain -eq 2 ]; then
            echo "False"
            pretrain="False"
        fi
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

if [ $model_used == "srcnn" ] || [ $model_used == "diff_unet" ]; then
    echo -n "Learning rate: "
    read lr
elif [ $model_used == "srgan" ]; then
    echo -n "Generator Learning rate: "
    read glr
    echo -n "Discriminator Learning rate: "
    read dlr
fi
echo -n "Batch size: "
read bs
echo -n "Epochs: "
read ep
echo -n "Device: (1) cpu (2) gpu -> "
read dev
if [ $dev -eq 1 ]; then
    echo "cpu"
    dev="cpu"
elif [ $dev -eq 2 ]; then
    echo "gpu"
    dev="gpu"
    echo -n "Distributed data parallel: (1) True (2) False -> "
    read ddp
    if [ $ddp -eq 1 ]; then
        echo "True"
        ddp="True"
    elif [ $ddp -eq 2 ]; then
        echo "False"
        ddp="False"
    fi
fi

echo -e "\nDefault values for other hyperparameters will be used...for more settings, change config.yaml file"
echo -e "\nTraining starting...\n"
if [ $exp_name == "super_resolution" ]; then
    if [ $model_used == 'srcnn' ]; then
        al="experiment.super_resolution.srcnn.train_param"
    elif [ $model_used == 'srgan' ]; then
        al="experiment.super_resolution.srgan.train_param"
    fi
elif [ $exp_name == "diffusion" ]; then
    if [ $model_used == 'diff_unet' ]; then
        al="experiment.diffusion.diff_unet.train_param"
    fi
fi

if [ $model_used == 'srcnn' ] || [ $model_used == 'diff_unet' ]; then
    params="experiment_name=$exp_name model_used=$model_used device=$dev $al.learning_rate=$lr $al.batch_size=$bs $al.epochs=$ep"
elif [ $model_used == 'srgan' ]; then
    params="experiment_name=$exp_name model_used=$model_used device=$dev $al.pretrain=$pretrain $al.gen_learning_rate=$glr $al.dis_learning_rate=$dlr $al.batch_size=$bs $al.epochs=$ep"
fi

if [ "$ddp" == "True" ]; then
    torchrun --standalone --nproc_per_node=gpu train.py $params
else
    python3.10 train.py $params
fi