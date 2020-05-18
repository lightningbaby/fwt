#! /bin/bash
#CUDA_VISIBLE_DEVICES=3,2,1 python3 train_encoder.py --method protonet --dataset fwt_train_wiki --testset fwt_val_pubmed --name gru2d_t1_t2_att_euc_gain_99 --warmup gru2d_t1_t2 --stop_epoch 100 | tee log_train_model/log_gru2d_t1_t2_att_euc_gain_encTrain99.txt &

CUDA_VISIBLE_DEVICES=3,2,1 python3 train_encoder.py --method protonet --dataset fwt_train_wiki --testset fwt_val_pubmed --name gru2d_t1_t2_att_euc_gain_79 --warmup gru2d_t1_t2_79 --stop_epoch 100 | tee log_train_model/log_gru2d_t1_t2_att_euc_gain_encTrain79.txt &

CUDA_VISIBLE_DEVICES=3,2,1 python3 train_encoder.py --method protonet --dataset fwt_train_wiki --testset fwt_val_pubmed --name gru2d_t1_t2_att_euc_gain_59 --warmup gru2d_t1_t2_59 --stop_epoch 100 | tee log_train_model/log_gru2d_t1_t2_att_euc_gain_encTrain59.txt &

CUDA_VISIBLE_DEVICES=3,2,1 python3 train_encoder.py --method protonet --dataset fwt_train_wiki --testset fwt_val_pubmed --name gru2d_t1_t2_att_euc_gain_39 --warmup gru2d_t1_t2_39 --stop_epoch 100 | tee log_train_model/log_gru2d_t1_t2_att_euc_gain_encTrain39.txt &



#CUDA_VISIBLE_DEVICES=3,2,1 python3 train_encoder.py --method protonet --dataset fwt_train_wiki --testset fwt_val_pubmed --name gru2d_t1_t2_att_mlp_gain_99 --warmup gru2d_t1_t2 --stop_epoch 100 | tee log_train_model/log_gru2d_t1_t2_att_mlp_gain_encTrain99.txt &

CUDA_VISIBLE_DEVICES=3,2,1 python3 train_encoder.py --method protonet --dataset fwt_train_wiki --testset fwt_val_pubmed --name gru2d_t1_t2_att_mlp_gain_79 --warmup gru2d_t1_t2_79 --stop_epoch 100 | tee log_train_model/log_gru2d_t1_t2_att_mlp_gain_encTrain79.txt &

CUDA_VISIBLE_DEVICES=3,2,1 python3 train_encoder.py --method protonet --dataset fwt_train_wiki --testset fwt_val_pubmed --name gru2d_t1_t2_att_mlp_gain_59 --warmup gru2d_t1_t2_59 --stop_epoch 100 | tee log_train_model/log_gru2d_t1_t2_att_mlp_gain_encTrain59.txt &

CUDA_VISIBLE_DEVICES=3,2,1 python3 train_encoder.py --method protonet --dataset fwt_train_wiki --testset fwt_val_pubmed --name gru2d_t1_t2_att_mlp_gain_39 --warmup gru2d_t1_t2_39 --stop_epoch 100 | tee log_train_model/log_gru2d_t1_t2_att_mlp_gain_encTrain39.txt 

