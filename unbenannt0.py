# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 14:01:44 2021

@author: User
"""

from data_wrapper_images import DataWrapperImages
#from tests import run_tests
import torch as T
import sys
import getopt

  
data_path = "E:\AML_data"
data_path_448 = "E:\AML_data_448x448"
use_cuda = True
try:
    opts, args = getopt.getopt(sys.argv[1:],"p:d:h",["path=", "downsampled_path=", "help"])
except getopt.GetoptError:
    print('main.py -p <path of data directory> -d <path of downsampled data directory>')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-p':
        data_path = arg
    if opt == '-d':
        data_path_448 = arg
data_path += "/"
data_path_448 += "/"

#region CUDA
if use_cuda and T.cuda.is_available():
    print("CUDA ENABLED")
    device = T.device("cuda")
else:                
    print("CUDA NOT AVAILABLE OR DISABLED")
    device = T.device("cpu")
#endregion

data_wrapper_images = DataWrapperImages(data_path, data_path_448)
data_wrapper_images.load_data_set()
#run_tests(device, data_wrapper_images)
