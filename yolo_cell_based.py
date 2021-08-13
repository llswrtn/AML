from yolo import *

class YoloCellBased(Yolo):
    """
    Cell based Yolo network.
    Each cell predicts the class independently.
    """
    def __init__(self, number_of_classes=4, boxes_per_cell=2, dropout_p=0.5, iou_mode=IOU_MODE_BOX_CENTER, activation_mode=ACTIVATION_MODE_LINEAR, clamp_box_dimensions=True):
        super(YoloCellBased, self).__init__(number_of_classes=number_of_classes, boxes_per_cell=boxes_per_cell, dropout_p=dropout_p, iou_mode=iou_mode, activation_mode=activation_mode, clamp_box_dimensions=clamp_box_dimensions)        
        print("init YoloCellBased")
      
    def init_helper_variables(self):
        self.values_per_cell = self.B*5+self.C
        self.out_layer_size = self.S*self.S*self.values_per_cell

    def apply_last_layer(self, x):
        #first without activation
        x = self.layer_30_full(x)
        self.print_debug("layer_30_full", x.size())
        x = T.reshape(x, (x.shape[0], self.S, self.S, self.values_per_cell))

        #in the paper, the last layer uses linear activation instead of leaky_relu
        if self.activation_mode == ACTIVATION_MODE_LINEAR:            
            return x
        #alternatively we tried sigmoid
        if self.activation_mode == ACTIVATION_MODE_SIGMOID:
            x = F.sigmoid(x)
            return x

        #in the case of linear activation, we also allow clamping of the coordinates to [0,1]
        if self.activation_mode == ACTIVATION_MODE_CLAMP_COORDINATES:
            x[:,:,:,self.coordinate_index_list] = T.clamp(x[:,:,:,self.coordinate_index_list], 0, 1) 
            return x
        #we also allow sigmoid only for coordinates and linear activation for all other values
        if self.activation_mode == ACTIVATION_MODE_SIGMOID_COORDINATES:            
            x[:,:,:,self.coordinate_index_list] = F.sigmoid(x[:,:,:,self.coordinate_index_list])
            return x

        print("ERROR UNKNOWN activation_mode", self.activation_mode)
        sys.exit(1)

    def to_separate_box_data(self, batch_element_index, input_tensor):
        """
        Called by prepare_data.
        Prepares the output of "forward" for use with to_converted_box_data.

        :param batch_element_index: the index in this batch.

        :param input_tensor: the input tensor with shape identical to the output of the forward method (batch_size, S, S, B*5+C).
        
        :returns separate_box_data: tensor in the shape (num_boxes, 5+C)
        """
        #extract data of the specified index in the batch
        data = input_tensor[batch_element_index]
        #reshape the data so that cells are in one dimension instead of two
        data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))

        #current shape: (S*S, values_per_cell)
        #the current shape describes B boxes per cell,
        #but we require the boxes to be separate.
        cells = self.S*self.S
        separate_box_data = T.zeros((cells*self.B, 5 + self.C))
        for i in range(self.B):
            #calculate the class indices from the number of boxes
            class_indices = T.arange(self.C)+(self.B*5)
            #calculate the first 5 values depending on i (the rest is copied in the next step)
            indices = T.arange(5 + self.C)+(i*5)
            #copy the class indices
            indices[-self.C:] = class_indices
            #extract data for the current box
            current_box_data = data[:, indices]
            #copy extracted data into 
            start = cells*i
            separate_box_data[start:start+cells,:] = current_box_data
        #print("separate_box_data", separate_box_data)
        return separate_box_data.to(self.device)  