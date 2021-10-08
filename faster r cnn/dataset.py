import ast
import numpy as np
import pandas as pd
import pydicom as dicom
import torch



class ImageLevelSiimCovid19Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, CLEAN_TRAIN_PATH):
        self.root = root

        # .csv under CLEAN_TRAIN_PATH contains all image paths with their bounding box annotations generated as output of notebook in preprocesing directory        
        self.CLEAN_TRAIN_PATH = CLEAN_TRAIN_PATH
        self.transforms = transforms
        
        
        self.imgs = pd.read_csv(CLEAN_TRAIN_PATH)['path'].tolist()
        self.annotations = pd.read_csv(CLEAN_TRAIN_PATH)['boxes'].tolist()
        
        self.study_class_labels = pd.read_csv(CLEAN_TRAIN_PATH)['study_label'].tolist()
        
    def __getitem__(self, idx):
        # load images and annotations
    
        img_path = self.imgs[idx]
        img = np.float32(dicom.dcmread(img_path).pixel_array)
        
        study_class_label = self.study_class_labels[idx]
        
        if pd.read_csv(self.CLEAN_TRAIN_PATH)['boxes'][idx] is np.NaN:
            boxes_dict = []
            
        else:
            boxes_dict = ast.literal_eval(self.annotations[idx])
            
        num_obj = len(boxes_dict)
    
        boxes = []
        for i in boxes_dict:

            x_min = i['x']
            y_min = i['y']
            x_max = i['x'] + i['width']
            y_max = i['y'] + i['height']
            boxes.append([x_min, y_min, x_max, y_max])
            
               
        #only one class label for the bounding boxes: opacity
        labels = torch.ones((num_obj,), dtype=torch.int64)
        
      
        #handle images without boxes in annotation
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
               
      
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        study_label = 0
        if study_class_label == 'Typical Appearance':
            study_label = 1
        if study_class_label == 'Indeterminate Appearance':
            study_label = 2
        if study_class_label == 'Atypical Appearance':
            study_label = 3

        study_label = torch.tensor([study_label])
        
        
        image_id = torch.tensor([idx])
        
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id        
        target["area"] = area
        
        target["study_label"] = study_label

        
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)        
        

        return img, target
        
        
    def __len__(self):
        return len(self.imgs)




class ResizedImageLevelSiimCovid19Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, CLEAN_TRAIN_PATH):
        self.root = root

        # .csv under CLEAN_TRAIN_PATH contains all image paths with their bounding box annotations generated as output of notebook in preprocesing directory        
        self.CLEAN_TRAIN_PATH = CLEAN_TRAIN_PATH
        self.transforms = transforms
        
        
        self.imgs = pd.read_csv(CLEAN_TRAIN_PATH)['resized_path'].tolist()
        self.annotations = pd.read_csv(CLEAN_TRAIN_PATH)['resized_boxes_list'].tolist()
        
        self.study_class_labels = pd.read_csv(CLEAN_TRAIN_PATH)['study_label'].tolist()
        
    def __getitem__(self, idx):
        # load images and annotations
    
        img_path = self.imgs[idx]
        img = np.load(img_path)
        
        study_class_label = self.study_class_labels[idx]
        
        if pd.read_csv(self.CLEAN_TRAIN_PATH)['boxes'][idx] is np.NaN:
            boxes_dict = []
            
        else:
            boxes_dict = ast.literal_eval(self.annotations[idx])
            
        num_obj = len(boxes_dict)
    
        boxes = boxes_dict
        '''
        for i in boxes_dict:

            x_min = i['x']
            y_min = i['y']
            x_max = i['x'] + i['width']
            y_max = i['y'] + i['height']
            boxes.append([x_min, y_min, x_max, y_max])
        '''    
               
        #only one class label for the bounding boxes: opacity
        labels = torch.ones((num_obj,), dtype=torch.int64)
        
      
        #handle images without boxes in annotation
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
               
      
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        study_label = 0
        if study_class_label == 'Typical Appearance':
            study_label = 1
        if study_class_label == 'Indeterminate Appearance':
            study_label = 2
        if study_class_label == 'Atypical Appearance':
            study_label = 3

        study_label = torch.tensor([study_label])
        
        
        image_id = torch.tensor([idx])
        
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id        
        target["area"] = area
        
        target["study_label"] = study_label

        
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)        
        

        return img, target
        
        
    def __len__(self):
        return len(self.imgs)

