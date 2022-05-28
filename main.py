import os
import argparse
from train import Trainer
import warnings
warnings.filterwarnings('ignore')
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--name', default='CBAMResnet50', type=str)
    parser.add_argument('--state_dir', default='./state', type=str)
    parser.add_argument('--log_dir', default='./log', type=str)
    parser.add_argument('--model', default='CBAMResnet50', type=str) # CBAMResnet50/resnet50

    # 调节参数的时候，可以调这三个参数。epoch、batch_size,learning rate
    parser.add_argument('--num_epoch', default=50, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)

    # 如果用cpu，则default里面改成cpu 用显卡则用cuda:0
    parser.add_argument('--gpu', default='cuda:0', type=str)

    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()
