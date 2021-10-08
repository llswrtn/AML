from data_wrapper_images import DataWrapperImages
from yolo import Yolo
import torch as T
import numpy as np
import pydicom as dicom
import matplotlib.pyplot as plt
import sys
import getopt
import os
from tqdm import tqdm
from PIL import Image

from skimage.transform import resize

if __name__ == "__main__":    
    data_path = "data"
    data_path_out = "data_448x448"
    use_cuda = True
    try:
        opts, args = getopt.getopt(sys.argv[1:],"i:o:h",["path_in=", "path_out=", "help"])
    except getopt.GetoptError:
        print('main.py -i <path of data input directory> -o <path of data output directory>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-i':
            data_path = arg
        if opt == '-o':
            data_path_out = arg
    data_path += "/"
    data_path_out += "/"
    data_path_train_in = data_path + "train/"
    data_path_train_out = data_path_out + "train/"

    print("data_path: "+data_path)
    print("data_path_out: "+data_path_out)
    
    dir_list = []
    file_list = []
    for dirpath, dirnames, filenames in os.walk(data_path_train_in):
        for filename in filenames:
            in_path = os.path.join(dirpath, filename)
            dir_list.append(dirpath)
            file_list.append(filename)

    for i in tqdm(range(len(file_list))):
        filename = file_list[i]
        in_path = os.path.join(dir_list[i], filename)
        dir_out = dir_list[i].replace(data_path, data_path_out)
        out_path = os.path.join(dir_out, filename.replace(".dcm", ".npy"))

        #if file is already converted, skip
        if os.path.exists(out_path):
            continue

        ds = dicom.dcmread(in_path)
        data = ds.pixel_array      
        IMG_PX_SIZE = 448

        #directory only needs to be created if it does not exist
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)
        
        resized_data = resize(data, (IMG_PX_SIZE, IMG_PX_SIZE), anti_aliasing=True)
        np.save(out_path, resized_data)