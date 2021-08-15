import os
import numpy as np
import pandas as pd
import pydicom as dicom
import matplotlib.pyplot as plt
import ast
import torch as T
from tqdm import tqdm

class DataWrapperImages:
    """
    Allows access to the train, validation, and test data via a batch index.
    - Call load_data_set to load the data.
    - Call split_train_validation_test to split the data for taining, validation, and testing.
    - Afterwards call get_train_batch, get_validation_batch, or get_test_batch
        to get all the required data of the respective batch.
    - The number of allowed batches can be obtained via get_num_train_batches, 
        get_num_validation_batches, or get_num_test_batches respectively. 
    """

    def __init__(self, data_path, data_path_448, keep_images_in_ram=True):
        print("data_path: ", data_path)
        print("data_path_448: ", data_path_448)
        print("keep_images_in_ram: ", keep_images_in_ram)
        self.data_path = data_path
        self.data_path_448 = data_path_448
        self.keep_images_in_ram = keep_images_in_ram
        self.data_frame = pd.read_csv (data_path+"train_image_level.csv")
        self.data_frame_study = pd.read_csv (data_path+"train_study_level.csv")

        self.num_total_images = len(self.data_frame.index)

        self.generate_path_lists()

    def generate_path_lists(self):
        print("generate path lists...")
        self.path_list = [None] * self.num_total_images
        self.path_448_list = [None] * self.num_total_images

        for i in tqdm(range(self.num_total_images)):
            self.path_list[i] = self.get_image_path(i, True)
            self.path_448_list[i] = self.get_image_path(i, False)
    
    def load_data_set(self):
        print("load data set...")
        self.load_data_set_image_size_list()
        self.load_data_set_luts()
        self.load_data_set_boxes()
        self.load_data_set_labels()   
        self.load_annotated_list()
        self.load_data_set_images()
        print("load data completed")
        #print(self.images_448)
        #print(self.images_448.shape)
        #print(self.ground_truth_boxes_list)
        #print(self.ground_truth_boxes_list.shape)

    def load_data_set_image_size_list(self):
        """
        Tries to load the list of image sizes from the data path.
        - If the file does exist, the sizes are loaded from the file
        - If the file does not exist, the image sizes are obtained
            from the dicom images. Afterwards the file is saved.
        """
        print("load data set image size list...")
        path_image_sizes = os.path.join(self.data_path, "image_sizes.npy")
        if os.path.exists(path_image_sizes):
            print("load image sizes from file")
            self.image_size_list = np.load(path_image_sizes)
        else:
            print("generate image sizes list from images...")
            self.image_size_list = np.empty((self.num_total_images, 2))
            for i in tqdm(range(self.num_total_images)):
                path = self.path_list[i]
                ds = dicom.dcmread(path)
                self.image_size_list[i,0] = ds.pixel_array.shape[1]
                self.image_size_list[i,1] = ds.pixel_array.shape[0]
            print("save image sizes list...")
            np.save(path_image_sizes, self.image_size_list)

    def load_data_set_images(self):
        if(self.keep_images_in_ram):
            print("load data set images...")
            self.images_448 = np.empty((self.num_total_images, 448, 448))
            for i in tqdm(range(self.num_total_images)):
                path = self.path_448_list[i]
                self.images_448[i] = np.load(path)
            print("loaded", self.images_448.shape, "images")

    def load_data_set_luts(self):
        print("load data set look up tables...")
        self.lut_image_id_to_image_index = {}
        self.lut_image_index_to_image_id = {}
        self.lut_image_id_to_study_id = {}
        self.lut_image_index_to_study_id = {}
        self.lut_study_id_to_list_image_id = {}
        self.lut_study_id_to_list_image_index = {}
        for image_index in tqdm(range(self.num_total_images)):
            image_id = self.data_frame.loc[image_index,"id"]
            study_id = self.data_frame.loc[image_index,"StudyInstanceUID"]
            self.lut_image_id_to_image_index[image_id] = image_index
            self.lut_image_index_to_image_id[image_index] = image_id
            self.lut_image_id_to_study_id[image_id] = study_id
            self.lut_image_index_to_study_id[image_index] = study_id
            #print(image_index, image_id, study_id)
            if study_id in self.lut_study_id_to_list_image_id:
                #append to entry
                self.lut_study_id_to_list_image_id[study_id].append(image_id)
                self.lut_study_id_to_list_image_index[study_id].append(image_index)
            else:
                #make new entry
                self.lut_study_id_to_list_image_id[study_id] = [image_id]
                self.lut_study_id_to_list_image_index[study_id] = [image_index]
        
    def load_annotated_list(self):
        """
        There are problems with the provided data. Some of the images have missing annotations.
        We therefore do not use images that have no bounding boxes if there is another image,
        of the same study, that does have bounding boxes.
        This is described in the following link:

        https://www.kaggle.com/c/siim-covid19-detection/discussion/246597

        Below is the full text from the above link:
            Our annotation team updated the labels for the test datasets (private and public)
            but the train labels remain the same.

            The updated test labels correct the problem regarding duplicates or extra images
            on some of the studies, which previously did not have bounding box information. 
            Bounding box information is now provided for duplicates or similar images that are 
            part of the same study for the public and private test datasets.

            However, regarding the train labels, due to time constraints, those labels were 
            kept the same and not updated. As such, in situations where there are 2 or more 
            images belonging to a study, we recommend only using the labels for the image with 
            the bounding boxes and disregard the other duplicate/similar images. These other 
            duplicate/similar images were likely not looked at by the annotators as we were 
            unaware of them during the initial annotation process. 
        """
        print("load annotated list...")
        self.annotated_list = []
        self.unannotated_list = []
        num_no_boxes_in_annotated_list = 0
        num_no_boxes = 0
        for image_index in range(self.num_total_images):
            boxes = self.ground_truth_boxes_list[image_index]
            label = self.labels[image_index]
            flag = True
            if boxes is None:
                num_no_boxes += 1
                #in the case of no existing boxes we have to check if other images of the same study contain boxes.                
                study_id = self.lut_image_index_to_study_id[image_index]
                list_image_index = self.lut_study_id_to_list_image_index[study_id]
                for _, other_image_index in enumerate(list_image_index):
                    other_boxes = self.ground_truth_boxes_list[other_image_index]
                    if not (other_boxes is None):
                        #another image (other_image_index) of the same study contain boxes, 
                        #which means that this image (image_index) was not annotated
                        flag = False
                #if no other box contains any boxes, our annotation containing no box is correct
                if flag:
                    num_no_boxes_in_annotated_list += 1
            #add the image_index to the list of annotated images if the image either:
            #   - contains boxes or 
            #   - does not contain boxes, but no other image in the associated study does not contain boxes either
            if flag:
                self.annotated_list.append(image_index)
            else:
                self.unannotated_list.append(image_index)
        
        num_labels_all = np.zeros(4)
        num_no_boxes_labels_all = np.zeros(4)
        for image_index in range(self.num_total_images):
            boxes = self.ground_truth_boxes_list[image_index]
            label = self.labels[image_index]
            num_labels_all += label
            if boxes is None:
                num_no_boxes_labels_all += label

        num_no_boxes_labels_annotated = np.zeros(4, dtype=np.intc)
        for _, image_index in enumerate(self.annotated_list):
            boxes = self.ground_truth_boxes_list[image_index]
            if boxes is None:
                label = self.labels[image_index]
                num_no_boxes_labels_annotated += label.astype(np.intc)

        num_no_boxes_labels_unannotated = np.zeros(4, dtype=np.intc)
        for _, image_index in enumerate(self.unannotated_list):
            boxes = self.ground_truth_boxes_list[image_index]
            if boxes is None:
                label = self.labels[image_index]
                num_no_boxes_labels_unannotated += label.astype(np.intc)

        print("num_total_images:", self.num_total_images)
        print("num_no_boxes:", num_no_boxes)
        print("num_no_boxes_in_annotated_list:", num_no_boxes_in_annotated_list)
        print("num_labels_all:", num_labels_all, "total:", np.sum(num_labels_all))
        print("num_no_boxes_labels_all", num_no_boxes_labels_all, "total:", np.sum(num_no_boxes_labels_all))
        print("num_no_boxes_labels_annotated", num_no_boxes_labels_annotated, "total:", np.sum(num_no_boxes_labels_annotated))
        print("num_no_boxes_labels_unannotated", num_no_boxes_labels_unannotated, "total:", np.sum(num_no_boxes_labels_unannotated))

        print("annotated_list:", len(self.annotated_list))
        print("unannotated_list:", len(self.unannotated_list))

    def load_data_set_boxes(self):
        """
        Loads ground truth boxes in the format (x1, y1, x2, y2) with 0 <= x1 < x2 and 0 <= y1 < y2
        with normalized coordinates.
        """
        print("load data set ground truth boxes...")
        self.ground_truth_boxes_list = [None] * self.num_total_images
        for i in tqdm(range(self.num_total_images)):
            boxes_string = self.data_frame.loc[i,"boxes"]
            if type(boxes_string) is str:             
                dict_list = ast.literal_eval(boxes_string)
                num_boxes = len(dict_list)
                boxes_data = np.empty((num_boxes, 4))
                for j in range(num_boxes):
                    x = dict_list[j]["x"] / self.image_size_list[i,0]
                    y = dict_list[j]["y"] / self.image_size_list[i,1]
                    w = dict_list[j]["width"] / self.image_size_list[i,0]
                    h = dict_list[j]["height"] / self.image_size_list[i,1]
                    boxes_data[j,0] = x
                    boxes_data[j,1] = y
                    boxes_data[j,2] = x+w
                    boxes_data[j,3] = y+h
                self.ground_truth_boxes_list[i] = boxes_data

    def load_data_set_labels(self):
        """
        Loads one hot encoded labels.
        """
        print("load data set labels...")
        path_labels = os.path.join(self.data_path, "labels.npy")
        if os.path.exists(path_labels):
            print("load labels from file")
            self.labels = np.load(path_labels)
        else:
            print("generate labels...")
            self.labels = np.empty((self.num_total_images, 4))
            for i in tqdm(range(self.num_total_images)):
                instance_id =  self.data_frame.loc[i,"StudyInstanceUID"] + "_study"           
                row = self.data_frame_study.loc[self.data_frame_study["id"] == instance_id]
                self.labels[i,0] = row["Negative for Pneumonia"]
                self.labels[i,1] = row["Typical Appearance"]
                self.labels[i,2] = row["Indeterminate Appearance"]
                self.labels[i,3] = row["Atypical Appearance"]
                #print("instance_id", instance_id)
                #print("self.labels[i]", self.labels[i])
            print("save label list...")
            np.save(path_labels, self.labels)

    def split_train_validation_test(self, p_train=0.8, p_validation=0.1, seed=1):
        #get number of indices for each set
        p_test = 1-p_train-p_validation
        num_train = int(p_train * self.num_total_images)
        num_validation = int(p_validation * self.num_total_images)
        num_test = self.num_total_images - num_train - num_validation

        #generate a random permutation of all indices
        np.random.seed(seed)
        all_indices = np.arange(self.num_total_images)
        all_indices = np.random.permutation(all_indices)

        self.train_indices = all_indices[0:num_train]
        self.validation_indices = all_indices[num_train:num_train+num_validation]
        self.test_indices = all_indices[num_train+num_validation:]
        
    def rearrange_train(self, seed):
        """
        Rearranges the train indices.
        """
        np.random.seed(seed)
        self.train_indices = np.random.permutation(self.train_indices)
        
    def get_num_train_batches(self, batch_size):
        num_batches = int(len(self.train_indices) / batch_size)
        return num_batches

    def get_num_validation_batches(self, batch_size):
        num_batches = int(len(self.validation_indices) / batch_size)
        return num_batches

    def get_num_test_batches(self, batch_size):
        num_batches = int(len(self.test_indices) / batch_size)
        return num_batches

    def get_train_batch(self, batch_index, batch_size, device):
        """
        Get the training data of the batch with the specified index.
        To get the number of batches, call get_num_train_batches

        :param batch_index: the index of the batch

        :param batch_size: the batch size

        :returns batch_images: a tensor with shape (batch_size, 448, 448)
        
        :returns batch_boxes: a list of batch_size tensors with variable shape
        
        returns batch_labels: a tensor with shape (batch_size, C)
        """
        return self.get_batch(batch_index, batch_size, self.train_indices, device)

    def get_validation_batch(self, batch_index, batch_size, device):
        """
        Get the validation data of the batch with the specified index.
        To get the number of batches, call get_num_validation_batches

        :param batch_index: the index of the batch

        :param batch_size: the batch size

        :returns batch_images: a tensor with shape (batch_size, 448, 448)
        
        :returns batch_boxes: a list of batch_size tensors with variable shape
        
        returns batch_labels: a tensor with shape (batch_size, C)
        """
        return self.get_batch(batch_index, batch_size, self.validation_indices, device)

    def get_test_batch(self, batch_index, batch_size, device):
        """
        Get the test data of the batch with the specified index.
        To get the number of batches, call get_num_test_batches

        :param batch_index: the index of the batch

        :param batch_size: the batch size

        :returns batch_images: a tensor with shape (batch_size, 448, 448)
        
        :returns batch_boxes: a list of batch_size tensors with variable shape
        
        returns batch_labels: a tensor with shape (batch_size, C)
        """
        return self.get_batch(batch_index, batch_size, self.test_indices, device)

    def get_batch(self, batch_index, batch_size, target, device):
        """
        Internal method used by get_train_batch, get_validation_batch, and get_test_batch
        """       
        start = batch_index*batch_size
        indices = target[start:start+batch_size]
        batch_images = self.get_images(indices, device)
        batch_images = T.reshape(batch_images, (batch_size, 1, 448, 448))
        batch_boxes = self.get_boxes(indices, device)
        batch_labels = self.get_labels(indices, device)
        return batch_images, batch_boxes, batch_labels

    def get_images(self, indices, device):
        """
        Returns a tensor containing the images associated with the indices.
        - If keep_images_in_ram is true, the images are obtained from self.images_448
        - otherwise each image is loaded from the associated file.
        """        
        batch_images = None
        if self.keep_images_in_ram:
            batch_images = self.images_448[indices]
        else:
            batch_size = indices.shape[0]
            batch_images = np.empty((batch_size,448,448))
            for i in range(batch_size):
                path = self.path_448_list[i]
                batch_images[i] = np.load(path)
        return T.tensor(batch_images, dtype=T.float32, device=device)

    def get_boxes(self, indices, device):
        """
        Returns a list of tensors containing the boxes associated with the indices.
        """  
        batch_size = indices.shape[0]      
        batch_boxes = [None] * batch_size
        for i in range(batch_size):
            index = indices[i]
            boxes = self.ground_truth_boxes_list[index]
            if not boxes is None:
                batch_boxes[i] = T.tensor(boxes, dtype=T.float32, device=device)
        return batch_boxes

    def get_labels(self, indices, device):
        """
        Returns a tensor of one hot encoded labels associated with the indices.
        """        
        batch_labels = self.labels[indices]
        return T.tensor(batch_labels, dtype=T.float32, device=device)

    def get_image_path(self, row_index, original=True):
        image_id = self.data_frame.loc[row_index,"id"].replace("_image", ".dcm")
        #boxes =  self.data_frame.loc[row_index,"boxes"]
        #label =  self.data_frame.loc[row_index,"label"]
        instance_id =  self.data_frame.loc[row_index,"StudyInstanceUID"]        
        path = self.data_path
        if not original:
            path = self.data_path_448 
            image_id = image_id.replace(".dcm", ".npy")
        path += "train/" + instance_id
        children = [f for f in os.scandir(path) if f.is_dir()]
        
        if len(children) == 0:
            print("error no children")
        else:
            for i, child in enumerate(children):
                potential_path = child.path + "/" + image_id
                if os.path.exists(potential_path):
                    return potential_path
            print("error no child with image_id")
        return ""

    def test_get_image_path(self):
        for i in range(6334):
            path = self.get_image_path(i)
            assert os.path.exists(path)

    def plot_row(self, row_index):
        path = self.get_image_path(row_index, original=True)
        ds = dicom.dcmread(path)
        plt.subplot(1, 2, 1)
        plt.imshow(ds.pixel_array)

        path = self.get_image_path(row_index, original=False)
        pixel_array = np.load(path)
        plt.subplot(1, 2, 2)
        plt.imshow(pixel_array)
        
        plt.show()