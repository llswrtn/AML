from data_wrapper_images import DataWrapperImages
from yolo import Yolo
from tests import run_tests
import torch as T
import numpy as np
import pydicom as dicom
import matplotlib.pyplot as plt
import sys
import getopt
import os
from PIL import Image

if __name__ == "__main__":    
    data_path = "data" 
    data_path_448 = "data_448x448"
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
    run_tests(device, data_wrapper_images)

    """
    path = data_wrapper_images.get_image_path(4438)
    #path = data_wrapper_images.get_image_path(0)
    path = "test.dcm"
    print(path)
    ds = dicom.dcmread(path)
    plt.imshow(ds.pixel_array)
    plt.show()
    """

    """
    resized_data = np.load("test.npy")
    plt.imshow(resized_data)
    plt.show()
    """


    #data_wrapper_images.plot_row(4438)
    #data_wrapper_images.plot_row(0)


