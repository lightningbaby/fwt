import os
os.system('CUDA_VISIBLE_DEVICE=0,1 python3 train_encoder.py --method baseline --dataset fwt_train_wiki --name gru2d_t1_t2  --stop_epoch 100 1>>log_encoder.txt')
os.system('CUDA_VISIBLE_DEVICE=0,1 python3 train_encoder.py --method protonet --dataset fwt_train_wiki --testset fwt_val_pubmed --name gru2d_t1_t2_att_euc_gain --warmup gru2d_t1_t2  --stop_epoch 100 1>>log_model.txt')
