device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
import numpy as np
import math
import random
import time
import datetime
from scipy.io import loadmat
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

import mne
from mne import Epochs, events_from_annotations, pick_types
from mne.channels import make_standard_montage, read_custom_montage

from torch import nn
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True



import torch
from torch import nn
from torch import Tensor
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat
import torch.nn.functional as F

from utils import numberClassChannel
from utils import load_data_evaluate
import xlsxwriter

import numpy as np
import pandas as pd

from pandas import ExcelWriter
from torch.autograd import Variable


from experiment import *
from TFCA_TransNet import *
def set_seed(seed_n):

    random.seed(seed_n)          
    np.random.seed(seed_n)        
    torch.manual_seed(seed_n)      

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(dirs = r"C:\Users",
         device ='device',              
         evaluate_mode = 'subject-dependent'
         heads=8,             # heads of MHA
         emb_size=40,         # token embding dim
         depth=6,             # Transformer encoder depth
         dataset_type='A',    # A->'BCI IV2a', B->'BCI IV2b'
         eeg1_f1=20,          # features of temporal conv
         eeg1_kernel_size=64, # kernel size of temporal conv
         eeg1_D=2,            # depth-wise conv 
         eeg1_pooling_size1=8,# p1
         eeg1_pooling_size2=8,# p2
         eeg1_dropout_rate=0.25,
         flatten_eeg1=600 ,   
         validate_ratio = 0.3,
         subject_id = None
         ):

    if not os.path.exists(dirs):
        os.makedirs(dirs)

    result_write_metric = ExcelWriter(dirs+"/result_metric.xlsx")
    
    result_metric_dict = {}
    y_true_pred_dict = { } 

    process_write = ExcelWriter(dirs+"/process_train.xlsx")
    pred_true_write = ExcelWriter(dirs+"/pred_true.xlsx")
    subjects_result = []
    best_epochs = []

    
    subject_seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 


    best = 0
    aver = 0
    
    subject_to_train = [subject_id-1] if subject_id else range(9)
    # 循环部分

    for i in  subject_to_train :
        starttime = datetime.datetime.now()
        
        # seed
        seed_n = subject_seeds[i]
        print('seed is ' + str(seed_n))
        # seed
        set_seed(seed_n)
        
        
        index_round = 0
        print('Subject %d' % (i + 1))
        exp = ExP(i + 1, 
                  data_dir=r"C:\Users", 
                  dir=dir,
                  result_name = './results',
                  epochs=600,
                  number_aug=2, 
                  number_seg=8, 
                  device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
                  evaluate_mode = evaluate_mode,
                  heads=heads, 
                  emb_size=emb_size,
                  depth=depth, 
                  dataset_type=dataset_type,
                  eeg1_f1 = eeg1_f1,
                  eeg1_kernel_size = eeg1_kernel_size,
                  eeg1_D = eeg1_D,
                  eeg1_pooling_size1 = eeg1_pooling_size1,
                  eeg1_pooling_size2 = eeg1_pooling_size2,
                  eeg1_dropout_rate = eeg1_dropout_rate,
                  flatten_eeg1 = flatten_eeg1,  
                  validate_ratio = 0.3
                  )

        bestAcc, averAcc, Y_true, Y_pred, df_process, best_epoch = exp.train()

        # Convert to NumPy format
        true_cpu = Y_true.cpu().numpy().astype(int)
        pred_cpu = Y_pred.cpu().numpy().astype(int)

        df_pred_true = pd.DataFrame({'pred': pred_cpu, 'true': true_cpu})
        df_pred_true.to_excel(pred_true_write, sheet_name=str(i))

        y_true_pred_dict[i] = df_pred_true

        # Performance Metrics
        accuracy, precision, recall, f1, kappa = calMetrics(true_cpu, pred_cpu)
        subject_result = {
            'accuracy': accuracy * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100,
            'kappa': kappa * 100
        }
        subjects_result.append(subject_result)
        df_process.to_excel(process_write, sheet_name=str(i))
        best_epochs.append(best_epoch)

        print(' THE BEST ACCURACY IS ' + str(bestAcc) + "\tkappa is " + str(kappa))

        endtime = datetime.datetime.now()
        print('subject %d duration: ' % i + str(endtime - starttime))

        best += bestAcc
        aver += averAcc
        if i == 0:  # Initialize yt (true labels) and yp (predicted labels)
            yt = Y_true
            yp = Y_pred
        else:
            yt = torch.cat((yt, Y_true))
            yp = torch.cat((yp, Y_pred))
            
    N_SUBJECT = 9

    best /= N_SUBJECT
    aver /= N_SUBJECT

    df_result = pd.DataFrame(subjects_result)
    mean = df_result.mean(axis=0)
    mean.name = 'mean'
    std = df_result.std(axis=0)
    std.name = 'std'
    df_result = pd.concat([df_result, pd.DataFrame(mean).T, pd.DataFrame(std).T])

    df_result.to_excel(result_write_metric, index=False)
    result_write_metric.close()
    process_write.close()
    pred_true_write.close()

    #result_write.write('**The average Best accuracy is: ' + str(best) + "\n")
    #result_write.write('The average Aver accuracy is: ' + str(aver) + "\n")
    print('**The average Best accuracy is: ' + str(best) + "\n")
    print('The average Aver accuracy is: ' + str(aver) + "\n") 

    return result_metric_dict





main()
 
print(datetime.datetime.now())
