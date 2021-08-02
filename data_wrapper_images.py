import os
import pandas as pd

class DataWrapperImages:

    def __init__(self):
        self.data_frame = pd.read_csv ("data/train_image_level.csv")

    def GetImagePath(self, row_index):
        image_id = self.data_frame.loc[row_index,"id"].replace("_image", ".dcm")
        #boxes =  self.data_frame.loc[row_index,"boxes"]
        #label =  self.data_frame.loc[row_index,"label"]
        instance_id =  self.data_frame.loc[row_index,"StudyInstanceUID"]        
        path = "data/train/" + instance_id
        children = [f for f in os.scandir(path) if f.is_dir()]
        
        if len(children) == 0:
            print("error no children")
        elif len(children) == 1:
            return (children[0].path + "/" + image_id)
        else:
            for i, child in enumerate(children):
                potential_path = child.path + "/" + image_id
                if os.path.exists(potential_path):
                    return (children[0].path + "/" + image_id)
            print("error no child with image_id")
        return ""