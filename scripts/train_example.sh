export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # set the visible GPUs
## Uncomment and set the following environment variables to use wandb
# export WANDB_KEY="YOUR_WANDB_KEY"
# export ENTITY="YOUR_ENTITY"
# export PROJECT="YOUR_PROJECT"
########################################################
GPUS=8 # number of GPUs to use

cd /path/to/SiT # <--- Change this to the path to the SiT repository

torchrun --nnodes=1 --nproc_per_node=${GPUS} \
         train.py \
         --model SiT-XL/2 \
         --data-path /path/to/imagenet/train \
         --results-dir /path/to/results \
         --global-batch-size 256 \
         --ckpt /path/to/checkpoint.pt \
        #  --wandb # <--- Uncomment this to use wandb