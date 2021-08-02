import pandas as pd

class DataWrapperImages:

    def __init__(self):
        self.data_frame = pd.read_csv ("data/train_image_level.csv")

    def GetImagePath(self, row_index):
        image_id = self.data_frame.loc[row_index,"id"]
        #boxes =  self.data_frame.loc[row_index,"boxes"]
        #label =  self.data_frame.loc[row_index,"label"]
        instance_id =  self.data_frame.loc[row_index,"StudyInstanceUID"]
        path = ("data/" + instance_id + "/" + image_id).replace("_image", ".dcm")
        return path