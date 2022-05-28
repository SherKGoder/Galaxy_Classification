import os
import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
from matplotlib import pyplot as plt
from tqdm import tqdm
from tensorflow.keras import utils
import h5py
from utils.log import Log
from random import sample
from resnet50 import resnet50
from CBAMResnet import CBAM_ResNet50
from dataset import DataGenerator
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import OneHotEncoder

writer = SummaryWriter(os.path.join('runs'))  # 创建一个folder存储需要记录的数据


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # logger 日志
        os.makedirs(self.args.log_dir, exist_ok=True)
        self.logger = Log(self.args.log_dir, self.args.name).get_logger()
        self.logger.info(json.dumps(vars(self.args)))

        # state checkpoint文件
        os.makedirs(self.args.state_dir, exist_ok=True)
        self.state_path = os.path.join(self.args.state_dir, self.args.name)
        if not os.path.exists(self.state_path):
            os.makedirs(self.state_path)

        # warm up
        # self.gradual_warmup_steps = [i * self.args.lr for i in torch.linspace(0.5, 2.0, 7)]
        # self.lr_decay_epochs = range(14, 47, self.args.lr_decay_step)

        self.logger.info("starting process data!")
        with h5py.File(os.path.join(args.data_path, 'Galaxy10_DECals.h5'), 'r') as F:
            images = np.array(F['images'])
            labels = np.array(F['ans'])

        # To convert the labels to categorical 10 classes
        labels = utils.to_categorical(labels, 10)
        # To convert to desirable type
        labels = labels.astype(np.float32)
        images = images.astype(np.float32)

        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder()
        indexlabels = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
        encoder = enc.fit(indexlabels)
        # encoded_labels = encoder.transform(labels)   #将索引值转换为onehot编码
        labels = encoder.inverse_transform(labels)  # 将onehot编码转换为索引值
        df = pd.DataFrame(data=labels)
        counts = df.value_counts().sort_index()

        half_images = []
        half_labels = []
        count = counts.values.ravel() // 2
        for i in range(len(images)):
            if count[labels[i]] != 0:
                count[labels[i]] = count[labels[i]] - 1;
                half_images.append(images[i])
                half_labels.append(labels[i][0])

        train_split_list = sample(range(len(half_labels)), int(len(half_labels) * 0.9))
        test_split_list = [item for item in range(len(half_labels)) if item not in train_split_list]
        train_images = np.array([half_images[i] for i in train_split_list])
        test_images = np.array([half_images[i] for i in test_split_list])
        train_labels = np.array([half_labels[i] for i in train_split_list])
        test_labels = np.array([half_labels[i] for i in test_split_list])

        features = ['Disturbed Galaxies', 'Merging Galaxies', 'Round Smooth Galaxies',
                    'In-between Round Smooth Galaxies', 'Cigar Shaped Smooth Galaxies', 'Barred Spiral Galaxies',
                    'Unbarred Tight Spiral Galaxies', 'Unbarred Loose Spiral Galaxies',
                    'Edge-on Galaxies without Bulge', 'Edge-on Galaxies with Bulge']

        # fig = plt.figure(figsize=(20, 20))
        # for i in range(25):
        #     plt.subplot(5, 5, i + 1)
        #     plt.imshow(train_images[i]/255)
        #     plt.title(features[int(train_labels[i])])
        #     fig.tight_layout(pad=3.0)
        # plt.show()

        self.train_dataset = DataGenerator(train_images, train_labels)
        self.test_dataset = DataGenerator(test_images, test_labels)
        self.trainloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=0,
                                      pin_memory=True)
        self.testloader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=0,
                                     pin_memory=True)

        self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        self.class_len = len(self.classes)
        self.logger.info("finish process data!")
        self.logger.info("训练集数量："+str(len(train_images)))
        self.logger.info("测试集数量："+str(len(test_images)))
        if args.model == "resnet50":
            print("resnet50")
            self.net = resnet50(num_classes=len(self.classes), pretrained=True).to(self.args.gpu)
        else:
            print("CBAMResnet50")
            self.net = CBAM_ResNet50(num_classes=len(self.classes)).to(self.args.gpu)
        # loss损失函数
        self.criterion = nn.CrossEntropyLoss()

        # optimizer 优化器
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr)

    def save_checkpoint(self, step):
        state = self.net.state_dict()
        # delete previous checkpoint
        for filename in os.listdir(self.state_path):
            if self.args.name in filename.split('.') and os.path.isfile(os.path.join(self.state_path, filename)):
                os.remove(os.path.join(self.state_path, filename))
        # save checkpoint
        torch.save(state, os.path.join(self.args.state_dir, self.args.name,
                                       self.args.name + '.' + str(step) + '.ckpt'))

    def save_model(self, best_step):
        os.rename(os.path.join(self.state_path, self.args.name + '.' + str(best_step) + '.ckpt'),
                  os.path.join(self.state_path, self.args.name + '.best'))

    def before_test_load(self):
        state = torch.load(os.path.join(self.state_path, self.args.name + '.best'), map_location=self.args.gpu)
        self.net.load_state_dict(state)

    def print_per_batch(self, prefix, metrics):
        self.logger.info(' ')
        self.logger.info('------------------------------------')
        self.logger.info('this batch')
        if prefix == 'train':
            self.logger.info('loss: {:.4f}'.format(np.mean(metrics.losses)))
        self.logger.info('macro: f1:{:.4f}\t recall:{:.4f}\t precision:{:.4f}'.format(metrics.macro_f1,
                                                                                      metrics.macro_recall,
                                                                                      metrics.macro_precision))
        self.logger.info('micro: f1:{:.4f}\t recall:{:.4f}\t precision:{:.4f}'.format(metrics.micro_f1,
                                                                                      metrics.micro_recall,
                                                                                      metrics.micro_precision))
        self.logger.info("------------------------------------")

    def print_per_epoch(self, prefix, metrics, epoch):
        self.logger.info('------------------------------------')
        if prefix == 'train':
            self.logger.info('epoch: {} | loss: {:.4f}'.format(epoch, np.mean(self.train_metrics.losses)))
        else:
            self.logger.info('final test results:')
        if prefix == 'train':
            self.logger.info('macro: f1:{:.4f}\t recall:{:.4f}\t precision:{:.4f}'.format(metrics.macro_f1,
                                                                                          metrics.macro_recall,
                                                                                          metrics.macro_precision))
            self.logger.info('micro: f1:{:.4f}\t recall:{:.4f}\t precision:{:.4f}'.format(metrics.micro_f1,
                                                                                          metrics.micro_recall,
                                                                                          metrics.micro_precision))
        else:
            self.logger.info('macro: f1:{:.4f}\t recall:{:.4f}\t precision:{:.4f}'.format(metrics.epoch_macro_f1,
                                                                                          metrics.epoch_macro_recall,
                                                                                          metrics.epoch_macro_precision))
            self.logger.info('micro: f1:{:.4f}\t recall:{:.4f}\t precision:{:.4f}'.format(metrics.epoch_micro_f1,
                                                                                          metrics.epoch_micro_recall,
                                                                                          metrics.epoch_micro_precision))
        self.logger.info("------------------------------------")

    def train(self):
        self.logger.info('start training')
        best_acc = 0.0
        best_epoch = 0
        # 开始训练
        for epoch in range(self.args.num_epoch):
            running_loss = 0.
            total = 0
            correct = 0
            train_acc = 0.
            # net.train()
            prefix = "train"
            tq = tqdm(self.trainloader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0, total=len(self.trainloader))
            for data in tq:
                inputs, labels = data
                inputs, labels = inputs.to(self.args.gpu), labels.to(self.args.gpu, dtype=torch.int64)
                # 梯度置为0
                self.optimizer.zero_grad()

                # 送入模型预测结果
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # 计算训练集上的准确度
                train_acc = correct / total

                # 计算损失函数
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                self.logger.info('epoch:%d, loss: %.4f, acc:%.4f' % (epoch + 1, loss.item(), train_acc))

                # break

            # 保存下目前训练集上表现最好的模型
            if train_acc > best_acc:
                best_epoch = epoch
                best_acc = train_acc
                self.save_checkpoint(epoch)
                self.logger.info('best_acc: {:.5f}'.format(best_acc))
            # break
            # 画loss和 train-acc的图
            writer.add_scalar('loss', loss, epoch)
            writer.add_scalar('train_acc', train_acc, epoch)
            self.evaluate(istest=True, epoch=epoch)

        self.logger.info('Finished Training')

        self.save_model(best_epoch)
        self.logger.info("finish training!")

        self.before_test_load()
        # 测试 测试集上的结果
        self.evaluate(istest=True, epoch=epoch)

    def evaluate(self, epoch, istest=False):
        correct = 0
        total = 0
        # net.eval()
        prefix = "test"
        tq = tqdm(self.testloader, desc='{}:'.format(prefix), ncols=0)
        with torch.no_grad():
            for data in tq:
                # 实现流程和train里面的一样
                images, labels = data
                images, labels = images.to(self.args.gpu), labels.to(self.args.gpu, dtype=torch.int64)
                # 下面四行，根据图像预测结果，然后计算准确度
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # break
            # 画val-acc的图
            writer.add_scalar('test_acc', 100 * correct / total, epoch)

        self.logger.info('Accuracy of the network on the all test images: %d %%' % (
                100 * correct / total))
        self.logger.info('testing finish!')
