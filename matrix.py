# Author: 无名的弗兰克 @ ChemDog
# https://github.com/Frank-the-Obscure/
# matrix for weibo data
# -*- coding: utf-8 -*-

import re
import jieba
import json
from sys import argv
import scipy.io as sio
import scipy.sparse as sp
import numpy as np

#script, filein_name = argv

def combine_matrix():
    
    X000 = [sio.loadmat(filein_name[:-4] + '0X000.mat')['X000'],
            sio.loadmat(filein_name[:-4] + '1X000.mat')['X000'],
            sio.loadmat(filein_name[:-4] + '2X000.mat')['X000']]

    X001 = [sio.loadmat(filein_name[:-4] + '0X001.mat')['X001'],
            sio.loadmat(filein_name[:-4] + '1X001.mat')['X001'],
            sio.loadmat(filein_name[:-4] + '2X001.mat')['X001']]

    X010 = [sio.loadmat(filein_name[:-4] + '0X010.mat')['X010'],
            sio.loadmat(filein_name[:-4] + '1X010.mat')['X010'],
            sio.loadmat(filein_name[:-4] + '2X010.mat')['X010']]

    X100 = [sio.loadmat(filein_name[:-4] + '0X100.mat')['X100'],
            sio.loadmat(filein_name[:-4] + '1X100.mat')['X100'],
            sio.loadmat(filein_name[:-4] + '2X100.mat')['X100']]
    

    X_000 = sp.vstack([X000[0],X000[1],X000[2]])
    X_001 = sp.vstack([X001[0],X001[1],X001[2]])
    X_010 = sp.vstack([X010[0],X010[1],X010[2]])
    X_100 = sp.vstack([X100[0],X100[1],X100[2]])
    print(X_000.shape)

    
    X_model_100 = sp.hstack([X_000,X_100])
    sio.savemat(filein_name[:-4] + 'X100-model.mat', {'X100':X_model_100})

    X_model_010 = sp.hstack([X_000,X_010])
    sio.savemat(filein_name[:-4] + 'X010-model.mat', {'X010':X_model_010})

    X_model_001 = sp.hstack([X_000,X_001])
    sio.savemat(filein_name[:-4] + 'X001-model.mat', {'X001':X_model_001})

    #print(X_model100.shape)

def combineX():
    matrix_list = ['08-12-cut1_Xl.mat', '08-12-cut2_Xl.mat']

    combined_list = []
    for i in matrix_list:
        print(i, sio.loadmat(i)['X'].shape)
        combined_list.append(sio.loadmat(i)['X'])

    combined_list = sp.vstack(combined_list)
    print(combined_list.shape)
    sio.savemat('08-12-cut_Xl.mat', {'X':combined_list})

def combiney():
    matrix_list = ['copy/7.txt-y001-model.mat', 'copy/8.txt-y001-model.mat',
                    'copy/9.txt-y001-model.mat', 'copy/10.txt-y001-model.mat']

    combined_list = []
    for i in matrix_list:
        combined_list.append(sio.loadmat(i)['X100'])

    combined_list = np.concatenate(combined_list, axis=0)
    #combined_list = sp.vstack(combined_list)
    print(combined_list.shape)
    sio.savemat(filein_name[:-4] + 'y001-7-10.mat', {'y001':combined_list})

#combine_matrix()
combineX()