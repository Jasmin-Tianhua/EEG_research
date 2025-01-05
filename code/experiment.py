import os
import numpy as np
import random
import torch
import datetime
from scipy.io import loadmat
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torch.autograd import Variable
import mne
from mne import Epochs, events_from_annotations, pick_types
from mne.channels import make_standard_montage
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from utils import *
from model_new import *


from utils import numberClassChannel
from utils import load_data_evaluate

  


class ExP():
    def __init__(self, nsub, data_dir, result_name, dir,
                 epochs=2000, 
                 number_aug=2,
                 number_seg=8, 
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 evaluate_mode = 'subject-dependent',
                 heads=4, 
                 emb_size=40,
                 depth=6, 
                 dataset_type='A',
                 eeg1_f1 = 20,
                 eeg1_kernel_size = 64,
                 eeg1_D = 2,
                 eeg1_pooling_size1 = 8,
                 eeg1_pooling_size2 = 8,
                 eeg1_dropout_rate = 0.3,
                 flatten_eeg1 = 600, 
                 validate_ratio = 0.2,
                 learning_rate = 0.001,
                 batch_size = 72,  
                 ):
        
        super(ExP, self).__init__()
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.lr = learning_rate
        self.b1 = 0.5
        self.b2 = 0.999
        self.n_epochs = epochs
        self.nSub = nsub
        self.number_augmentation = number_aug
        self.number_seg = number_seg
        self.root = data_dir
        self.heads=heads
        self.emb_size=emb_size
        self.depth=depth
        self.result_name = result_name
        self.dir = dir
        self.evaluate_mode = evaluate_mode
        self.validate_ratio = validate_ratio
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_write = open(f"results/log_subject{self.nSub}.txt", "w")

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor
        self.criterion_cls = torch.nn.CrossEntropyLoss().to(device)

        self.number_class, self.number_channel = numberClassChannel(self.dataset_type)
        self.model = EEGTransformer(
             heads=self.heads, 
             emb_size=self.emb_size,
             depth=self.depth, 
            database_type=self.dataset_type, 
            eeg1_f1=eeg1_f1, 
            eeg1_D=eeg1_D,
            eeg1_kernel_size=eeg1_kernel_size,
            eeg1_pooling_size1 = eeg1_pooling_size1,
            eeg1_pooling_size2 = eeg1_pooling_size2,
            eeg1_dropout_rate = eeg1_dropout_rate,
            eeg1_number_channel = self.number_channel,
            flatten_eeg1 = flatten_eeg1,  
            ).to(device)
        #self.model = nn.DataParallel(self.model, device_ids=gpus) 
        self.model = self.model.to(device)
        self.model_filename = self.result_name + '/model_{}.pth'.format(self.nSub)

    # Segmentation and Reconstruction (S&R) data augmentation
    def interaug(self, timg, label):  
        aug_data = []
        aug_label = []
        number_records_by_augmentation = self.number_augmentation * int(self.batch_size / self.number_class)
        number_segmentation_points = 1000 // self.number_seg
        for clsAug in range(self.number_class):
            cls_idx = np.where(label == clsAug + 1)
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]
            
            tmp_aug_data = np.zeros((number_records_by_augmentation, 1, self.number_channel, 1000))
            for ri in range(number_records_by_augmentation):
                for rj in range(self.number_seg):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], self.number_seg)
                    tmp_aug_data[ri, :, :, rj * number_segmentation_points:(rj + 1) * number_segmentation_points] = \
                        tmp_data[rand_idx[rj], :, :, rj * number_segmentation_points:(rj + 1) * number_segmentation_points]

            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:number_records_by_augmentation])
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data).to(device)
        aug_data = aug_data.float()
        aug_label = torch.from_numpy(aug_label-1).to(device)
        aug_label = aug_label.long()
        return aug_data, aug_label


    def fft_features(self, time_signal):
        # FFT 
        fft_result = torch.fft.fft(time_signal, dim=-1)  # FFT along time axis
        amplitude = torch.abs(fft_result)  # amplitude
        phase = torch.angle(fft_result)  # phase
     
        concatenated_features = torch.cat([time_signal, amplitude, phase], dim=-1)
        return concatenated_features



    def standardize(self, data):

        part_size = 1000 #Adjust this according to the sampling frequency and time.
        standardized_parts = []
        
        for i in range(data.size(-1) // part_size):
            part = data[..., i * part_size:(i + 1) * part_size]
            mean = torch.mean(part)
            std = torch.std(part)
            standardized_part = (part - mean) / (std + 1e-6)  #  std 
            standardized_parts.append(standardized_part)

        standardized_data = torch.cat(standardized_parts, dim=-1)
        return standardized_data



    def get_source_data(self):
        (self.train_data,    # (batch, channel, length)
         self.train_label, 
         self.test_data, 
         self.test_label) = load_data_evaluate(self.root, self.dataset_type, self.nSub, mode_evaluate=self.evaluate_mode)

        self.train_data = np.expand_dims(self.train_data, axis=1)  # (288, 1, 22, 1000)
        self.train_label = np.transpose(self.train_label)  

        self.allData = self.train_data
        self.allLabel = self.train_label[0]  

        shuffle_num = np.random.permutation(len(self.allData))
        # print("len(self.allData):", len(self.allData))
        self.allData = self.allData[shuffle_num, :, :, :]  # (288, 1, 22, 1000)
        # print("shuffle_num", shuffle_num)
        # print("self.allLabel", self.allLabel)
        self.allLabel = self.allLabel[shuffle_num]


        print('-'*20, "train size：", self.train_data.shape, "test size：", self.test_data.shape)
        # self.test_data = np.transpose(self.test_data, (2, 1, 0))
        self.test_data = np.expand_dims(self.test_data, axis=1)
        self.test_label = np.transpose(self.test_label)

        self.testData = self.test_data
        self.testLabel = self.test_label[0]


        return self.allData, self.allLabel, self.testData, self.testLabel







    def train(self):
        img, label, test_data, test_label = self.get_source_data()

        img = torch.from_numpy(img)
        label = torch.from_numpy(label - 1)
        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label - 1)

        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)

        # Optimizers
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        
        test_data = Variable(test_data.type(self.Tensor))
        test_data = self.fft_features(test_data)
        test_data = self.standardize(test_data)
        test_label = Variable(test_label.type(self.LongTensor))

        bestAcc = 0
        best_epoch = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0


        process_data = []

        for e in range(self.n_epochs):
            self.model.train()
            for i, (img, label) in enumerate(self.dataloader):
                img = Variable(img.to(device).type(self.Tensor))
                label = Variable(label.to(device).type(self.LongTensor))

                # data augmentation
                #print("Before Augmentation - Mean:", np.mean(self.allData), "Std:", np.std(self.allData))
                aug_data, aug_label = self.interaug(self.allData, self.allLabel)
                #print("After Augmentation - Mean:", np.mean(aug_data.cpu().numpy()), "Std:", np.std(aug_data.cpu().numpy()))
                img = torch.cat((img, aug_data))
                label = torch.cat((label, aug_label))
                #fft
                img = self.fft_features(img)
                img = self.standardize(img)

                tok, outputs = self.model(img)
                loss = self.criterion_cls(outputs, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # test process
            if (e + 1) % 1 == 0:
                self.model.eval()
                Tok, Cls = self.model(test_data)

                loss_test = self.criterion_cls(Cls, test_label)
                y_pred = torch.max(Cls, 1)[1]
                acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))

                print('Epoch:', e,
                    '  Train loss: %.6f' % loss.detach().cpu().numpy(),
                    '  Test loss: %.6f' % loss_test.detach().cpu().numpy(),
                    '  Train accuracy %.6f' % train_acc,
                    '  Test accuracy is %.6f' % acc)

                self.log_write.write(str(e) + "    " + str(acc) + "\n")

                process_data.append({
                    'epoch': e,
                    'train_loss': loss.detach().cpu().numpy(),
                    'test_loss': loss_test.detach().cpu().numpy(),
                    'train_acc': train_acc,
                    'test_acc': acc
                })
 
                num += 1
                averAcc += acc
                if acc > bestAcc:
                    bestAcc = acc
                    best_epoch = e  
                    Y_true = test_label
                    Y_pred = y_pred

        averAcc /= num
        print('The average accuracy is:', averAcc)
        print('The best accuracy is:', bestAcc)
        self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")

        df_process = pd.DataFrame(process_data)

        return bestAcc, averAcc, Y_true, Y_pred, df_process, best_epoch
