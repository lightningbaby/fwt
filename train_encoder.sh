#! /bin/bash
CUDA_VISIBLE_DEVICES=2,1,0 python3 train_encoder.py --method baseline --dataset fwt_train_wiki --name gru2d_t1_t2 --stop_epoch 100

CUDA_VISIBLE_DEVICES=2,1,0 python3 train_encoder.py --method protonet --dataset fwt_train_wiki --testset fwt_val_pubmed --name gru2d_t1_t2_att_euc_gain_99 --warmup gru2d_t1_t2 --stop_epoch 100 | tee log_train_model/log_gru2d_t1_t2_att_euc_gain_encTrain99.txt &

CUDA_VISIBLE_DEVICES=2,1,0 python3 train_encoder.py --method protonet --dataset fwt_train_wiki --testset fwt_val_pubmed --name gru2d_t1_t2_att_mlp_gain_99 --warmup gru2d_t1_t2 --stop_epoch 100 | tee log_train_model/log_gru2d_t1_t2_att_mlp_gain_encTrain99.txt &
