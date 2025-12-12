export CUDA_VISIBLE_DEVICES=0 

cd /path/to/SiT # <--- Change this to the path to the SiT repository

python sample.py \
--model SiT-XL/2 \
--vae ema \
--image-size 256 \
--num-classes 1000 \
--cfg-scale 4.0 \
--class-conditioning \
--num-sampling-steps 100 \
--seed 42 \
--ckpt /path/to/checkpoint.pt