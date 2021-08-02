from data_wrapper_images import DataWrapperImages


if __name__ == "__main__":
    data_wrapper_images = DataWrapperImages()
    path = data_wrapper_images.GetImagePath(0)
    print(path)

