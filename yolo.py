from basic_network import *

class Yolo(BasicNetwork):
    """
    Neural Network test
    """
    def __init__(self, number_of_classes=4, boxes_per_cell=2, dropout_p=0.5):
        super(Yolo, self).__init__()        
        print("init Yolo")
        self.leaky_slope = 0.1  
        self.number_of_classes = number_of_classes
        self.boxes_per_cell = boxes_per_cell
        self.dropout_p = dropout_p
        #alternative names for easier use
        self.B = boxes_per_cell
        self.C = number_of_classes        
        self.S = 7
        #helper variables
        self.values_per_cell = self.B*5+self.C

        self.generate_grid_data()

        #setup layers
      

        #region COLUMN 0
        #Conv. Layer 7x7x64-s-2
        self.layer_0_conv = nn.Conv2d(
            in_channels=3, 
            out_channels=64, 
            kernel_size=7, 
            stride=2, 
            padding=3)
        #Maxpool Layer 2x2-s-2
        self.layer_1_maxpool = nn.MaxPool2d(
            kernel_size=2, 
            stride=2, 
            padding=0)
        #endregion

        #region COLUMN 1
        #Conv. Layer 3x3x192
        self.layer_2_conv = nn.Conv2d(
            in_channels=64, 
            out_channels=192, 
            kernel_size=3, 
            stride=1, 
            padding=1)
        #Maxpool Layer 2x2-s-2
        self.layer_3_maxpool = nn.MaxPool2d(
            kernel_size=2, 
            stride=2, 
            padding=0)
        #endregion

        #region COLUMN 2
        #Conv. Layer 1x1x128
        self.layer_4_conv = nn.Conv2d(
            in_channels=192, 
            out_channels=128, 
            kernel_size=1, 
            stride=1, 
            padding=0)
        #Conv. Layer 3x3x256
        self.layer_5_conv = nn.Conv2d(
            in_channels=128, 
            out_channels=256, 
            kernel_size=3, 
            stride=1, 
            padding=1)
        #Conv. Layer 1x1x256
        self.layer_6_conv = nn.Conv2d(
            in_channels=256, 
            out_channels=256, 
            kernel_size=1, 
            stride=1, 
            padding=0)
        #Conv. Layer 3x3x512
        self.layer_7_conv = nn.Conv2d(
            in_channels=256, 
            out_channels=512, 
            kernel_size=3, 
            stride=1, 
            padding=1)
        #Maxpool Layer 2x2-s-2
        self.layer_8_maxpool = nn.MaxPool2d(
            kernel_size=2, 
            stride=2, 
            padding=0)
        #endregion

        #region COLUMN 3
        #start loop
        #iteration 1/4
        #Conv. Layer 1x1x256
        self.layer_9_conv = nn.Conv2d(
            in_channels=512, 
            out_channels=256, 
            kernel_size=1, 
            stride=1, 
            padding=0)
        #Conv. Layer 3x3x512
        self.layer_10_conv = nn.Conv2d(
            in_channels=256, 
            out_channels=512, 
            kernel_size=3, 
            stride=1, 
            padding=1)
        #iteration 2/4
        #Conv. Layer 1x1x256
        self.layer_11_conv = nn.Conv2d(
            in_channels=512, 
            out_channels=256, 
            kernel_size=1, 
            stride=1, 
            padding=0)
        #Conv. Layer 3x3x512
        self.layer_12_conv = nn.Conv2d(
            in_channels=256, 
            out_channels=512, 
            kernel_size=3, 
            stride=1, 
            padding=1)
        #iteration 3/4
        #Conv. Layer 1x1x256
        self.layer_13_conv = nn.Conv2d(
            in_channels=512, 
            out_channels=256, 
            kernel_size=1, 
            stride=1, 
            padding=0)
        #Conv. Layer 3x3x512
        self.layer_14_conv = nn.Conv2d(
            in_channels=256, 
            out_channels=512, 
            kernel_size=3, 
            stride=1, 
            padding=1)
        #iteration 4/4
        #Conv. Layer 1x1x256
        self.layer_15_conv = nn.Conv2d(
            in_channels=512, 
            out_channels=256, 
            kernel_size=1, 
            stride=1, 
            padding=0)
        #Conv. Layer 3x3x512
        self.layer_16_conv = nn.Conv2d(
            in_channels=256, 
            out_channels=512, 
            kernel_size=3, 
            stride=1, 
            padding=1)
        #end loop
        #Conv. Layer 1x1x512
        self.layer_17_conv = nn.Conv2d(
            in_channels=512, 
            out_channels=512, 
            kernel_size=1, 
            stride=1, 
            padding=0)
        #Conv. Layer 3x3x1024
        self.layer_18_conv = nn.Conv2d(
            in_channels=512, 
            out_channels=1024, 
            kernel_size=3, 
            stride=1, 
            padding=1)
        #Maxpool Layer 2x2-s-2
        self.layer_19_maxpool = nn.MaxPool2d(
            kernel_size=2, 
            stride=2, 
            padding=0)
        #endregion

        #region COLUMN 4
        #start loop
        #iteration 1/2
        #Conv. Layer 1x1x512
        self.layer_20_conv = nn.Conv2d(
            in_channels=1024, 
            out_channels=512, 
            kernel_size=1, 
            stride=1, 
            padding=0)
        #Conv. Layer 3x3x1024
        self.layer_21_conv = nn.Conv2d(
            in_channels=512, 
            out_channels=1024, 
            kernel_size=3, 
            stride=1, 
            padding=1)
        #iteration 2/2
        #Conv. Layer 1x1x512
        self.layer_22_conv = nn.Conv2d(
            in_channels=1024, 
            out_channels=512, 
            kernel_size=1, 
            stride=1, 
            padding=0)
        #Conv. Layer 3x3x1024
        self.layer_23_conv = nn.Conv2d(
            in_channels=512, 
            out_channels=1024, 
            kernel_size=3, 
            stride=1, 
            padding=1)
        #end loop
        #Conv. Layer 3x3x1024
        self.layer_24_conv = nn.Conv2d(
            in_channels=1024, 
            out_channels=1024, 
            kernel_size=3, 
            stride=1, 
            padding=1)
        #Conv. Layer 3x3x1024-s-2
        self.layer_25_conv = nn.Conv2d(
            in_channels=1024, 
            out_channels=1024, 
            kernel_size=3, 
            stride=2, 
            padding=1)
        #endregion

        #region COLUMN 5
        #Conv. Layer 3x3x1024
        self.layer_26_conv = nn.Conv2d(
            in_channels=1024, 
            out_channels=1024, 
            kernel_size=3, 
            stride=1, 
            padding=1)
        #Conv. Layer 3x3x1024
        self.layer_27_conv = nn.Conv2d(
            in_channels=1024, 
            out_channels=1024, 
            kernel_size=3, 
            stride=1, 
            padding=1)
        #endregion

        #region COLUMN 6
        #Conn Layer
        self.layer_28_full = nn.Linear(7*7*1024, 4069)
        #endregion

        #region DROPOUT
        self.layer_29_dropout = nn.Dropout(dropout_p)
        #endregion

        #region COLUMN 7
        #Conn Layer
        self.layer_30_full = nn.Linear(4069, self.S*self.S*(self.values_per_cell))
        #endregion

    def generate_grid_data(self):
        """
        Generates a grid with shape (num_boxes, 6) with the following values for each box:
        grid x index, grid y index, cell min x, cell max x, cell min y, cell max y
        """
        x = T.arange(self.S)
        y = T.arange(self.S)
        grid_x, grid_y  = T.meshgrid(x, y)
        flat_x = T.reshape(grid_x, (self.S*self.S, 1))
        flat_y = T.reshape(grid_y, (self.S*self.S, 1))
        data = T.cat((flat_x, flat_y), dim=1)
        min_x = data[:,0] / self.S
        max_x = (data[:,0]+1) / self.S
        min_y = data[:,1] / self.S
        max_y = (data[:,1]+1) / self.S
        #append dimension for concatenating
        min_x = T.reshape(min_x, (*min_x.shape, 1))
        max_x = T.reshape(max_x, (*max_x.shape, 1))
        min_y = T.reshape(min_y, (*min_y.shape, 1))
        max_y = T.reshape(max_y, (*max_y.shape, 1))
        data = T.cat((data, min_x, max_x, min_y, max_y), dim=1)

        stacked_data = data
        for i in range(self.B-1):            
            stacked_data = T.cat((stacked_data, data), dim=0)

        self.grid_data = data
        self.stacked_grid_data = stacked_data

    def forward(self, x):     
        #print("forward:", x.size())
        x = F.leaky_relu(self.layer_0_conv(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_0_conv", x.size())
        x = F.leaky_relu(self.layer_1_maxpool(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_1_maxpool", x.size())
        x = F.leaky_relu(self.layer_2_conv(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_2_conv", x.size())
        x = F.leaky_relu(self.layer_3_maxpool(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_3_maxpool", x.size())
        x = F.leaky_relu(self.layer_4_conv(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_4_conv", x.size())
        x = F.leaky_relu(self.layer_5_conv(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_5_conv", x.size())
        x = F.leaky_relu(self.layer_6_conv(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_6_conv", x.size())
        x = F.leaky_relu(self.layer_7_conv(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_7_conv", x.size())
        x = F.leaky_relu(self.layer_8_maxpool(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_8_maxpool", x.size())
        x = F.leaky_relu(self.layer_9_conv(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_9_conv", x.size())
        x = F.leaky_relu(self.layer_10_conv(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_10_conv", x.size())
        x = F.leaky_relu(self.layer_11_conv(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_11_conv", x.size())
        x = F.leaky_relu(self.layer_12_conv(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_12_conv", x.size())
        x = F.leaky_relu(self.layer_13_conv(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_13_conv", x.size())
        x = F.leaky_relu(self.layer_14_conv(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_14_conv", x.size())
        x = F.leaky_relu(self.layer_15_conv(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_15_conv", x.size())
        x = F.leaky_relu(self.layer_16_conv(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_16_conv", x.size())
        x = F.leaky_relu(self.layer_17_conv(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_17_conv", x.size())
        x = F.leaky_relu(self.layer_18_conv(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_18_conv", x.size())
        x = F.leaky_relu(self.layer_19_maxpool(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_19_maxpool", x.size())
        x = F.leaky_relu(self.layer_20_conv(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_20_conv", x.size())
        x = F.leaky_relu(self.layer_21_conv(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_21_conv", x.size())
        x = F.leaky_relu(self.layer_22_conv(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_22_conv", x.size())
        x = F.leaky_relu(self.layer_23_conv(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_23_conv", x.size())
        x = F.leaky_relu(self.layer_24_conv(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_24_conv", x.size())
        x = F.leaky_relu(self.layer_25_conv(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_25_conv", x.size())
        x = F.leaky_relu(self.layer_26_conv(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_26_conv", x.size())
        x = F.leaky_relu(self.layer_27_conv(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_27_conv", x.size())
        #tensor needs to be flattened to 2d to work with fully connected layer
        #first dimension for batch, second for the flattened part
        x = T.flatten(x, start_dim=1)
        x = F.leaky_relu(self.layer_28_full(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_28_full", x.size())
        x = F.leaky_relu(self.layer_29_dropout(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_29_dropout", x.size())
        
        #last layer uses relu instead of leaky_relu
        x = F.relu(self.layer_30_full(x))
        self.print_debug("layer_30_full", x.size())

        #reshape tensor
        x = T.reshape(x, (x.shape[0], self.S, self.S, self.values_per_cell))
        return x

    def to_separate_box_data(self, batch_index, input_tensor):
        """
        Prepares the output of "forward" for use with "non_max_suppression".

        :param batch_index: the index in this batch.

        :param input_tensor: the input tensor with shape identical to the output of the forward method (batch_size, S, S, B*5+C).
        
        :returns separate_box_data: tensor in the shape (num_boxes, 5+C)
        """
        #extract data of the specified index in the batch
        data = input_tensor[batch_index]
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
        return separate_box_data    

    def to_converted_box_data(self, separate_box_data):
        """
        Converts boxes from (ccenterx, ccentery, w, h) to (x1, y1, x2, y2)

        :param input_tensor: the input tensor with shape (num_boxes, 5+C) as provided by to_separate_box_data.
        box coordinates are represented as (ccenterx, ccentery, w, h)

        :returns converted_box_data: tensor in the shape (num_boxes, 5+C)
        box coordinates are represented as (x1, y1, x2, y2) with 0 <= x1 < x2 and 0 <= y1 < y2
        """
        converted_box_data = separate_box_data
        #center coordinates are relative to the grid cells
        #interpolate inside the grid cells to get coordinates relative to image
        center_x = T.lerp(
            self.stacked_grid_data[:,2], 
            self.stacked_grid_data[:,3], 
            separate_box_data[:,0])
        center_y = T.lerp(
            self.stacked_grid_data[:,4], 
            self.stacked_grid_data[:,5], 
            separate_box_data[:,1])

        #calculate min and max coordinates using the center coordinates relative to the image
        #and half width/height
        w_half = separate_box_data[:,2] / 2
        h_half = separate_box_data[:,3] / 2
        converted_box_data[:,0] = center_x - w_half
        converted_box_data[:,1] = center_y - h_half
        converted_box_data[:,2] = center_x + w_half
        converted_box_data[:,3] = center_y + h_half
        return converted_box_data

    def non_max_suppression(self, batch_index, input_tensor, iou_threshold=0.5, score_threshold=0.5):
        """
        Applies nms to one image of the batch.

        :param batch_index: the index in this batch.

        :param input_tensor: the input tensor with shape identical to the output of the forward method
        (batch_size, S, S, B*5+C).
        """
        #torchvision.ops.nms requires one tensor for the boxes, and one tensor for the scores.
        #split cells into B separate boxes
        separate_box_data = self.to_separate_box_data(batch_index, input_tensor)
        #convert boxes from (ccenterx, ccentery, w, h) to (x1, y1, x2, y2) 
        converted_box_data = self.to_converted_box_data(separate_box_data)
        boxes = converted_box_data[:,[0,1,2,3]]#(x1, y1, x2, y2) with 0 <= x1 < x2 and 0 <= y1 < y2
        scores = converted_box_data[:,4]
        #pre filtering via score threshold
        filter_score_threshold = scores > score_threshold
        filter_indices = T.arange(self.S*self.S*self.B)[filter_score_threshold]
        #apply non maximum suppression with pre filtered data
        keep_indices = torchvision.ops.nms(boxes[filter_score_threshold], 
            scores[filter_score_threshold], iou_threshold=iou_threshold)
        #get the unfiltered indices
        correct_indices = filter_indices[keep_indices]
        #get the converted box data for those indices
        filtered_converted_box_data = converted_box_data[correct_indices]

        filtered_grid_data = self.stacked_grid_data[correct_indices]
        return correct_indices, filtered_converted_box_data, filtered_grid_data