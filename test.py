import os
import h5py
import torch
import argparse
import numpy as np
from tqdm import tqdm
from scipy import interp
from utils.log import Log
from dataset import DataGenerator
from itertools import cycle
from resnet50 import resnet50
from CBAMResnet import CBAM_ResNet50
import matplotlib.pyplot as plt
from random import sample
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    average_precision_score, roc_curve, precision_recall_curve, auc
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(os.path.join('runs'))  # 创建一个folder存储需要记录的数据

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--name', default='CBAMResnet50', type=str)
    parser.add_argument('--state_dir', default='./state', type=str)
    parser.add_argument('--log_dir', default='./log', type=str)
    parser.add_argument('--model', default='CBAMResnet50', type=str)
    parser.add_argument('--model-path', default="./state/CBAMResnet50/CBAMResnet50.19.ckpt", type=str)
    parser.add_argument('--epoch', default=20, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--gpu', default='cuda:0', type=str)
    args = parser.parse_args()

    with h5py.File(os.path.join(args.data_path, 'Galaxy10_DECals.h5'), 'r') as F:
        images = np.array(F['images'])
        labels = np.array(F['ans'])

    test_labels = labels.astype(np.float32)
    test_images = images.astype(np.float32)

    # To get the images and labels from file
    test_dataset = DataGenerator(test_images, test_labels)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                            pin_memory=True)
    # classes = ('0', '1', '2', '3','4', '5', '6', '7', '8', '9')
    classes = ('Disturbed Galaxies', 'Merging Galaxies', 'Round Smooth Galaxies', 'In-between Round Smooth Galaxies',
               'Cigar Shaped Smooth Galaxies', 'Barred Spiral Galaxies',
               'Unbarred Tight Spiral Galaxies', 'Unbarred Loose Spiral Galaxies', 'Edge-on Galaxies without Bulge',
               'Edge-on Galaxies with Bulge')
    class_len = len(classes)

    logger = Log(args.log_dir, args.name).get_logger()
    correct = 0
    total = 0
    # net.eval()
    prefix = "test"
    tq = tqdm(testloader, desc='{}:'.format(prefix), ncols=0)
    if args.model == "resnet50":
        print("resnet50")
        net = resnet50(num_classes=len(classes), pretrained=True).to(args.gpu)
    else:
        print("CBAMResnet50")
        net = CBAM_ResNet50(num_classes=len(classes)).to(args.gpu)
    net.load_state_dict(torch.load(args.model_path))
    class_correct = list(0. for i in range(class_len))
    class_total = list(0. for i in range(class_len))
    label_all = []
    pred_all = []
    flag = 0
    with torch.no_grad():
        for data in tq:
            inputs, labels = data
            inputs, labels = inputs.to(args.gpu), labels.to(args.gpu, dtype=torch.int64)
            # 根据图像预测结果，然后计算准确度
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            print(predicted, labels)
            flag += 1
