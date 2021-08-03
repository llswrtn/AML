import os
import pandas as pd

class DataWrapperImages:

    def __init__(self, data_path):
        print("data_path: "+data_path)
        self.data_path = data_path
        self.data_frame = pd.read_csv (data_path+"train_image_level.csv")

    def GetImagePath(self, row_index):
        image_id = self.data_frame.loc[row_index,"id"].replace("_image", ".dcm")
        #boxes =  self.data_frame.loc[row_index,"boxes"]
        #label =  self.data_frame.loc[row_index,"label"]
        instance_id =  self.data_frame.loc[row_index,"StudyInstanceUID"]        
        path = self.data_path+"train/" + instance_id
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

    def TestGetImagePath(self):
        for i in range(6334):
            path = self.GetImagePath(i)
            assert os.path.exists(path)