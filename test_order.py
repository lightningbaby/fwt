import argparse
import os

parser = argparse.ArgumentParser(description='test script %s')
parser.add_argument('--name', default='Name',help='model name')

params=parser.parse_args()

name = params.name
root = 'log_test'
root_path = os.path.join(root,name)
epoch=['-1','99','79','59','39','19']
out_path=[]
os.mkdir(root_path)
for i in range(len(epoch)):
 # e=str(epoch[i])
  out_path.append(os.path.join(root_path,(epoch[i]+'.txt')))
#os.makedirs(out_path[0])
#print(out_path)
# os.mkdirs(out_path)

os.system('python3 re_test.py --method protonet --name %s --dataset fwt_val_pubmed --save_epoch -1 | tee %s'%(name,out_path[0]))
os.system('python3 re_test.py --method protonet --name %s --dataset fwt_val_pubmed --save_epoch 99 | tee %s'%(name,out_path[1]))
os.system('python3 re_test.py --method protonet --name %s --dataset fwt_val_pubmed --save_epoch 79 | tee %s'%(name,out_path[2]))
os.system('python3 re_test.py --method protonet --name %s --dataset fwt_val_pubmed --save_epoch 59 | tee %s'%(name,out_path[3]))
os.system('python3 re_test.py --method protonet --name %s --dataset fwt_val_pubmed --save_epoch 39 | tee %s'%(name,out_path[4]))
os.system('python3 re_test.py --method protonet --name %s --dataset fwt_val_pubmed --save_epoch 19 | tee %s'%(name,out_path[5]))

