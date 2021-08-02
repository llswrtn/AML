from data_wrapper_images import DataWrapperImages
import pydicom as dicom
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_wrapper_images = DataWrapperImages()
    path = data_wrapper_images.GetImagePath(0)
    print(path)
    ds = dicom.dcmread(path)
    plt.imshow(ds.pixel_array)
    plt.show()
