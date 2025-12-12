# Flow-Matching Training Example for Trinity On-Board
This repo is for Trinity users at CMU who want to get on-board to the cluster by training something. Before proceeding, make sure you have walked through the Trinity cluster Doc and prepare the training data properly.

The backbone model used is SiT and the implementation is also based on the original SiT repo. You can visit their repo at: [https:/https://github.com/willisma/SiT/tree/main/github.com/rosinality/swin-image-transformer](https://github.com/willisma/SiT/tree/main)

## Setup
First clone the repository
``` bash
git clone https://github.com/nickgong1/Trinity-on-board.git
```

There is an environment file for you to setup the conda environment
```bash
conda env create -f environment.yml
conda activate SiT
```
## Train from Scratch
There is a minimal script for you to train the SiT model from scratch using the ImageNet data you prepared. Run the following script and modify arguments accordingly.
```bash
cd YOUR_PATH_TO_TRINITY_ON_BOARD
bash ./scripts/train_example.sh
```
## Sampling
To sample images, you can download the pretrained checkpoints from original SiT repo or you can load your own checkpoints. A minimal script to run sampling is provided and make sure you change the arguments accordingly.
```bash
cd YOUR_PATH_TO_TRINITY_ON_BOARD
bash ./scripts/sample_example.sh
```


