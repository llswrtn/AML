from data_wrapper_images import DataWrapperImages
from yolo import Yolo
import torch as T
import numpy as np
import pydicom as dicom
import matplotlib.pyplot as plt
import sys
import getopt
import os
from PIL import Image

def TestYolo():
    number_of_classes = 4
    yolo = Yolo(number_of_classes).to(device)

    data = np.zeros((3,448,448))
    data = data.reshape(1, *data.shape)
    input_tensor = T.tensor(data, dtype=T.float32, device=device)
    output_tensor = yolo(input_tensor)
    #print(output_tensor)
    print(output_tensor.shape)
    #yolo.save("save.pt")#--> 1GB

def RunTests():
    print("running tests...")
    data_wrapper_images.TestGetImagePath()
    TestYolo()
    print("tests completed")


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
    RunTests()

    """
    path = data_wrapper_images.GetImagePath(4438)
    #path = data_wrapper_images.GetImagePath(0)
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
    data_wrapper_images.plot_row(0)


