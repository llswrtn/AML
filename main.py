from data_wrapper_images import DataWrapperImages
import pydicom as dicom
import matplotlib.pyplot as plt
import sys
import getopt
import os

def RunTests():
    print("running tests...")
    data_wrapper_images.TestGetImagePath()
    print("tests completed")

if __name__ == "__main__":    
    data_path = "data"
    try:
        opts, args = getopt.getopt(sys.argv[1:],"p:h",["path=", "help"])
    except getopt.GetoptError:
        print('main.py -p <path of data directory>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-p':
            data_path = arg
    data_path += "/"

    data_wrapper_images = DataWrapperImages(data_path)
    RunTests()

    path = data_wrapper_images.GetImagePath(4438)
    #path = data_wrapper_images.GetImagePath(0)
    print(path)
    ds = dicom.dcmread(path)
    plt.imshow(ds.pixel_array)
    plt.show()
