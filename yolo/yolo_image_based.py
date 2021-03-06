from yolo import *

class YoloImageBased(Yolo):
    """
    Image based Yolo network.
    Only one class prediction per image.
    """
    def __init__(self, number_of_classes=4, boxes_per_cell=2, dropout_p=0.5, architecture=ARCHITECTURE_DEFAULT, iou_mode=IOU_MODE_BOX_CENTER, activation_mode=ACTIVATION_MODE_LINEAR, clamp_box_dimensions=True, prediction_method=PREDICTION_MAX, prediction_method_no_boxes=PREDICTION_NO_BOXES_FALLBACK_IGNORE_CONFIDENCE, allow_classification_error=True, allow_classification_error_no_boxes=True):
        super(YoloImageBased, self).__init__(number_of_classes=number_of_classes, boxes_per_cell=boxes_per_cell, dropout_p=dropout_p, architecture=architecture, iou_mode=iou_mode, activation_mode=activation_mode, clamp_box_dimensions=clamp_box_dimensions, prediction_method=prediction_method, prediction_method_no_boxes=prediction_method_no_boxes, allow_classification_error=allow_classification_error, allow_classification_error_no_boxes=allow_classification_error_no_boxes)   
        print("init YoloImageBased")
        self.is_image_based = True
      
    def init_helper_variables(self):
        self.values_per_cell = self.B*5
        self.out_layer_size = self.S*self.S*self.values_per_cell + self.C

    def apply_last_layer(self, x):
        #first without activation
        x = self.layer_last_full(x)
        self.print_debug("layer_last_full", x.size())

        #in the paper, the last layer uses linear activation instead of leaky_relu
        if self.activation_mode == ACTIVATION_MODE_LINEAR:
            return x
        #alternatively we tried sigmoid
        if self.activation_mode == ACTIVATION_MODE_SIGMOID:
            x = F.sigmoid(x)
            return x

        #in the case of linear activation, we also allow clamping of the coordinates to [0,1]
        if self.activation_mode == ACTIVATION_MODE_CLAMP_COORDINATES:
            y = x[:, :self.C]
            x = x[:, self.C:]
            x = T.reshape(x, (x.shape[0], self.S, self.S, self.values_per_cell))
            x[:,:,:,self.coordinate_index_list] = T.clamp(x[:,:,:,self.coordinate_index_list], 0, 1) 
            x = T.reshape(x, (x.shape[0], self.S*self.S*self.values_per_cell))  
            x = T.cat((y, x), dim=1)
            return x
        #we also allow sigmoid only for coordinates and linear activation for all other values
        if self.activation_mode == ACTIVATION_MODE_SIGMOID_COORDINATES:
            y = x[:, :self.C]
            x = x[:, self.C:]
            x = T.reshape(x, (x.shape[0], self.S, self.S, self.values_per_cell))
            x[:,:,:,self.coordinate_index_list] = F.sigmoid(x[:,:,:,self.coordinate_index_list])
            x = T.reshape(x, (x.shape[0], self.S*self.S*self.values_per_cell))  
            x = T.cat((y, x), dim=1)
            return x

        print("ERROR UNKNOWN activation_mode", self.activation_mode)
        sys.exit(1)

    def to_separate_box_data(self, batch_element_index, input_tensor):
        """
        Called by prepare_data.
        Prepares the output of "forward" for use with to_converted_box_data.

        :param batch_element_index: the index in this batch.

        :param input_tensor: the input tensor with shape identical to the output of the forward method (batch_size, C+S*S*B*5).
        
        :returns separate_box_data: tensor in the shape (num_boxes, 5+C)
        """
        cells = self.S*self.S
        #extract data of the specified index in the batch
        data = input_tensor[batch_element_index]
        y = data[:self.C]
        x = data[self.C:]
        #reshape the data so that cells are in one dimension instead of two
        x = T.reshape(x, (cells, self.values_per_cell))

        #current shape: (S*S, values_per_cell)
        #the current shape describes B boxes per cell,
        #but we require the boxes to be separate.
        separate_box_data = T.zeros((cells*self.B, 5 + self.C))
        for i in range(self.B):
            #calculate the first 5 values depending on i
            indices = T.arange(5)+(i*5)
            start = cells*i
            separate_box_data[start:start+cells,:5] = x[:, indices]
            #classes are the same for each box
            separate_box_data[start:start+cells,5:] = y
        return separate_box_data.to(self.device)  
