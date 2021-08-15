from basic_network import *

"""
iou_mode:
Each ground truth box has exactly one prediction that is responsible for it.

- IOU_MODE_ALL: 
    In this mode, the responsible box is the box with highest IOU of all S*S*B boxes.

- IOU_MODE_BOX_CENTER: 
    This is probably the better version.
    In this mode, the responsible box is the box with highest IOU of all B boxes that are in the grid cell
    where the center of the ground truth box falls into.
"""
IOU_MODE_ALL = 0
IOU_MODE_BOX_CENTER = 1

"""
activation_mode:
Each ground truth box has exactly one prediction that is responsible for it.

- ACTIVATION_MODE_LINEAR: 
    In this mode, linear activation is used for all values of the last layer.

- ACTIVATION_MODE_SIGMOID: 
    In this mode, sigmoid is used for all values of the last layer.

- ACTIVATION_MODE_CLAMP_COORDINATES: 
    In this mode, linear activation is used for all values of the last layer.
    Additionally coordinates of the last layer are clamped between [0,1].

- ACTIVATION_MODE_SIGMOID_COORDINATES: 
    In this mode, sigmoid is used for coordinate values of the last layer.
    All other values of the last layer use linear activation.
"""
ACTIVATION_MODE_LINEAR = 0
ACTIVATION_MODE_SIGMOID = 1
ACTIVATION_MODE_CLAMP_COORDINATES = 2
ACTIVATION_MODE_SIGMOID_COORDINATES = 3

"""
architecture:
Determines which layer architecture is used.

- ARCHITECTURE_DEFAULT: 
    24 convolution YOLO network described in the paper

- ARCHITECTURE_FAST: 
    9 convolution YOLO network mentioned in the paper
"""
ARCHITECTURE_DEFAULT = 0
ARCHITECTURE_FAST = 1

"""
prediction_method:
Determines how multiple class predictions (from multiple boxes) result in a single class prediction for the image.

- PREDICTION_MAX: 
    use the highest class specific confidence found

- PREDICTION_MAX_MEAN: 
    use the highest mean class specific confidence found
"""
PREDICTION_MAX = 0
PREDICTION_MAX_MEAN = 1

class Yolo(BasicNetwork):
    """
    Base class for the yolo networks
    """
    def __init__(self, number_of_classes=4, boxes_per_cell=2, dropout_p=0.5, architecture=ARCHITECTURE_DEFAULT, iou_mode=IOU_MODE_BOX_CENTER, activation_mode=ACTIVATION_MODE_LINEAR, clamp_box_dimensions=True, prediction_method=PREDICTION_MAX):
        super(Yolo, self).__init__()        
        print("init Yolo")
        self.leaky_slope = 0.1  
        self.number_of_classes = number_of_classes
        self.boxes_per_cell = boxes_per_cell
        self.dropout_p = dropout_p
        self.architecture = architecture
        self.prediction_method = prediction_method
        self.iou_mode = iou_mode
        self.activation_mode = activation_mode
        self.clamp_box_dimensions = clamp_box_dimensions  
        #alternative names for easier use
        self.B = boxes_per_cell
        self.C = number_of_classes        
        self.S = 7
        #helper variables
        self.init_helper_variables()   
        #setup layers depending on the selected yolo architecture
        if self.architecture == ARCHITECTURE_DEFAULT:
            self.generate_layers_default()
        elif self.architecture == ARCHITECTURE_FAST:
            self.generate_layers_fast()
        else:
            print("unknown architecture: ", self.architecture)
            sys.exit(1)

    def generate_layers_default(self):
        print("generate_layers_default")
        #region COLUMN 0
        #Conv. Layer 7x7x64-s-2
        self.layer_0_conv = nn.Conv2d(
            in_channels=1, 
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
        self.layer_29_dropout = nn.Dropout(self.dropout_p)
        #endregion

        #region COLUMN 7
        #Conn Layer
        self.layer_last_full = nn.Linear(4069, self.out_layer_size)
        #endregion

    def generate_layers_fast(self):
        """
        We assume kernel_size=3 for all convolutional layers
        """
        #1. convolution
        #Conv. Layer 7x7x16
        self.layer_0_conv = nn.Conv2d(
            in_channels=1, 
            out_channels=16, 
            kernel_size=3, 
            stride=1, 
            padding=1)
        #Maxpool Layer 2x2-s-2
        self.layer_1_maxpool = nn.MaxPool2d(
            kernel_size=2, 
            stride=2, 
            padding=0)

        #2. convolution
        #Conv. Layer 3x3x32
        self.layer_2_conv = nn.Conv2d(
            in_channels=16, 
            out_channels=32, 
            kernel_size=3, 
            stride=1, 
            padding=1)
        #Maxpool Layer 2x2-s-2
        self.layer_3_maxpool = nn.MaxPool2d(
            kernel_size=2, 
            stride=2, 
            padding=0)

        #3. convolution
        #Conv. Layer 3x3x64
        self.layer_4_conv = nn.Conv2d(
            in_channels=32, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1)
        #Maxpool Layer 2x2-s-2
        self.layer_5_maxpool = nn.MaxPool2d(
            kernel_size=2, 
            stride=2, 
            padding=0)

        #4. convolution
        #Conv. Layer 3x3x128
        self.layer_6_conv = nn.Conv2d(
            in_channels=64, 
            out_channels=128, 
            kernel_size=3, 
            stride=1, 
            padding=1)
        #Maxpool Layer 2x2-s-2
        self.layer_7_maxpool = nn.MaxPool2d(
            kernel_size=2, 
            stride=2, 
            padding=0)

        #5. convolution
        #Conv. Layer 3x3x256
        self.layer_8_conv = nn.Conv2d(
            in_channels=128, 
            out_channels=256, 
            kernel_size=3, 
            stride=1, 
            padding=1)
        #Maxpool Layer 2x2-s-2
        self.layer_9_maxpool = nn.MaxPool2d(
            kernel_size=2, 
            stride=2, 
            padding=0)

        #6. convolution
        #Conv. Layer 3x3x512
        self.layer_10_conv = nn.Conv2d(
            in_channels=256, 
            out_channels=512, 
            kernel_size=3, 
            stride=1, 
            padding=1)
        #Maxpool Layer 2x2-s-2
        self.layer_11_maxpool = nn.MaxPool2d(
            kernel_size=2, 
            stride=2, 
            padding=0)

        #7. convolution
        #Conv. Layer 3x3x1024
        self.layer_12_conv = nn.Conv2d(
            in_channels=512, 
            out_channels=1024, 
            kernel_size=3, 
            stride=1, 
            padding=1)

        #8. convolution
        #Conv. Layer 3x3x1024
        self.layer_13_conv = nn.Conv2d(
            in_channels=1024, 
            out_channels=1024, 
            kernel_size=3, 
            stride=1, 
            padding=1)

        #9. convolution
        #Conv. Layer 3x3x1024
        self.layer_14_conv = nn.Conv2d(
            in_channels=1024, 
            out_channels=1024, 
            kernel_size=3, 
            stride=1, 
            padding=1)

        #Conn Layers
        self.layer_15_full = nn.Linear(7*7*1024, 256)
        self.layer_16_full = nn.Linear(256, 4069)
        self.layer_last_full = nn.Linear(4069, self.out_layer_size)

    def initialize(self, device):
        self.device = device
        self.generate_grid_data()
        self.generate_coordinate_index_list()

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

        self.grid_data = data.to(self.device)
        self.stacked_grid_data = stacked_data.to(self.device)

    def generate_coordinate_index_list(self):
        box_indices = T.arange(self.B)
        box_indices = T.reshape(box_indices, (*box_indices.shape, 1))
        coord_offsets = T.arange(2)
        coord_offsets = T.reshape(coord_offsets, (1, *coord_offsets.shape))

        self.coordinate_index_list = 5 * box_indices + coord_offsets
        self.coordinate_index_list = T.flatten(self.coordinate_index_list).to(self.device)

    def forward(self, x):     
        #apply layers depending on the selected yolo architecture
        if self.architecture == ARCHITECTURE_DEFAULT:
            x = self.apply_layers_default(x)
        elif self.architecture == ARCHITECTURE_FAST:
            x = self.apply_layers_fast(x)
        else:
            print("unknown architecture: ", self.architecture)
            sys.exit(1)
        #last layer:
        x = self.apply_last_layer(x)
        self.print_debug("return x.shape", x.shape)
        return x

    def apply_layers_default(self, x):
        """
        Applies the default yolo layers except the last layer
        """
        x = self.layer_0_conv(x)#no activation because next layer is maxpool, followed by activation
        self.print_debug("layer_0_conv", x.size())
        x = F.leaky_relu(self.layer_1_maxpool(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_1_maxpool", x.size())
        x = self.layer_2_conv(x)#no activation because next layer is maxpool, followed by activation
        self.print_debug("layer_2_conv", x.size())
        x = F.leaky_relu(self.layer_3_maxpool(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_3_maxpool", x.size())
        x = F.leaky_relu(self.layer_4_conv(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_4_conv", x.size())
        x = F.leaky_relu(self.layer_5_conv(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_5_conv", x.size())
        x = F.leaky_relu(self.layer_6_conv(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_6_conv", x.size())
        x = self.layer_7_conv(x)#no activation because next layer is maxpool, followed by activation
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
        x = self.layer_18_conv(x)#no activation because next layer is maxpool, followed by activation
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
        return x

    def apply_layers_fast(self, x):
        x = self.layer_0_conv(x)#no activation because next layer is maxpool, followed by activation
        self.print_debug("layer_0_conv", x.size())
        x = F.leaky_relu(self.layer_1_maxpool(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_1_maxpool", x.size())
        x = self.layer_2_conv(x)#no activation because next layer is maxpool, followed by activation
        self.print_debug("layer_2_conv", x.size())
        x = F.leaky_relu(self.layer_3_maxpool(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_3_maxpool", x.size())
        x = self.layer_4_conv(x)#no activation because next layer is maxpool, followed by activation
        self.print_debug("layer_4_conv", x.size())
        x = F.leaky_relu(self.layer_5_maxpool(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_5_maxpool", x.size())
        x = self.layer_6_conv(x)#no activation because next layer is maxpool, followed by activation
        self.print_debug("layer_6_conv", x.size())
        x = F.leaky_relu(self.layer_7_maxpool(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_7_maxpool", x.size())
        x = self.layer_8_conv(x)#no activation because next layer is maxpool, followed by activation
        self.print_debug("layer_8_conv", x.size())
        x = F.leaky_relu(self.layer_9_maxpool(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_9_maxpool", x.size())
        x = self.layer_10_conv(x)#no activation because next layer is maxpool, followed by activation
        self.print_debug("layer_10_conv", x.size())
        x = F.leaky_relu(self.layer_11_maxpool(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_11_maxpool", x.size())

        x = F.leaky_relu(self.layer_12_conv(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_12_conv", x.size())
        x = F.leaky_relu(self.layer_13_conv(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_13_conv", x.size())
        x = F.leaky_relu(self.layer_14_conv(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_14_conv", x.size())

        #tensor needs to be flattened to 2d to work with fully connected layer
        #first dimension for batch, second for the flattened part
        x = T.flatten(x, start_dim=1)
        x = F.leaky_relu(self.layer_15_full(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_15_full", x.size())
        x = F.leaky_relu(self.layer_16_full(x), negative_slope=self.leaky_slope)
        self.print_debug("layer_16_full", x.size())
        return x

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
        #linear activation allows negative width and height, which is not supported.
        if self.clamp_box_dimensions:
            w_half = T.clamp(w_half, 0, 1)
            h_half = T.clamp(h_half, 0, 1)

        converted_box_data[:,0] = center_x - w_half
        converted_box_data[:,1] = center_y - h_half
        converted_box_data[:,2] = center_x + w_half
        converted_box_data[:,3] = center_y + h_half
        return converted_box_data

    def prepare_data(self, batch_element_index, forward_result):
        """
        Converts the results of the forward method for use with the loss function and non max suppression.

        :param batch_element_index: the index in this batch.

        :param forward_result: the output of the forward method
        (batch_size, S, S, B*5+C).
        """
        #split cells into B separate boxes
        separate_box_data = self.to_separate_box_data(batch_element_index, forward_result)
        #convert boxes from (ccenterx, ccentery, w, h) to (x1, y1, x2, y2) 
        converted_box_data = self.to_converted_box_data(separate_box_data)
        return converted_box_data.to(self.device)

    def non_max_suppression(self, converted_box_data, iou_threshold=0.5, score_threshold=0.5):
        """
        Applies nms to one image of the batch.

        :param converted_box_data: tensor obtained by prepare_data with shape (num_boxes, 5+C)
        """
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
        #get the grid data for those indices
        filtered_grid_data = self.stacked_grid_data[correct_indices]
        return correct_indices, filtered_converted_box_data, filtered_grid_data

    def get_responsible_indices(self, converted_box_data, ground_truth_boxes):
        """
        Identify which boxes are responsible. Uses self.iou_mode to decide responsibility.
        See top of this file for explanations of the different modes.

        :param batch_index: the index in this batch.

        :param input_tensor: the input tensor with shape identical to the output of the forward method
        (batch_size, S, S, B*5+C).

        :returns responsible_indices: the box indices responsible for each ground truth box
        tensor in the shape (num_ground_truth_boxes)

        :returns responsible_indices_1: 1 or 0 for each (box, ground_truth_box) pair, 1 if the box is responsible
        for the specific ground truth box of the pair
        tensor in the shape (num_boxes, num_ground_truth_boxes)

        :returns responsible_indices_any_1: 1 or 0 for each box, 1 if the box is responsible for any of the ground truth boxes
        tensor in the shape (num_boxes, 1)

        :returns responsible_indices_noobj_1: 0 or 1 for each box, 1 if the box is NOT responsible for any of the ground truth boxes
        tensor in the shape (num_boxes, 1)
        the paper mentions that they 'decrease the loss from confidence predictions for boxes that don't contain objects',
        so we interpret this as the inverted responsible_indices_any_1, but we are unsure if this is correct
        """
        responsible_indices = T.empty((ground_truth_boxes.shape[0]), dtype=T.long)
        if self.iou_mode == IOU_MODE_ALL:
            iou = torchvision.ops.box_iou(converted_box_data[:,0:4], ground_truth_boxes)
            responsible_indices = T.argmax(iou, dim=0)
        elif self.iou_mode == IOU_MODE_BOX_CENTER:
            #get the cells responsible for each ground truth box
            responsible_cells_mask, responsible_cells_1, responsible_cells_indices = self.get_responsible_cells(ground_truth_boxes)
            print("responsible_cells_mask", responsible_cells_mask)
            print("responsible_cells_mask.shape", responsible_cells_mask.shape)
            #extend the responsible cell indices to all B boxes
            num_box_indices = T.arange(self.B, device=self.device)
            num_box_indices = T.reshape(num_box_indices, (*num_box_indices.shape, 1))
            cell_indices_extended = num_box_indices * self.S*self.S + responsible_cells_indices
            print("cell_indices_extended", cell_indices_extended)
            #apply iou for each ground truth box
            for i in range(ground_truth_boxes.shape[0]):
                #get and reshape the ground truth box
                ground_truth_box = ground_truth_boxes[i]
                ground_truth_box = T.reshape(ground_truth_box, (1, *ground_truth_box.shape))
                print("ground_truth_box", ground_truth_box)
                #get the indices of all B boxes that are associated with the responsible cell of the current ground truth box
                cell_indices_extended_i = cell_indices_extended[:,i]
                #get only those B boxes
                filtered_boxes = converted_box_data[cell_indices_extended_i,0:4]
                #apply iou to those B boxes and the current ground_truth_box
                iou = torchvision.ops.box_iou(filtered_boxes, ground_truth_box)
                print("iou", iou)
                #get the filtered index 
                responsible_index_j = T.argmax(iou, dim=0)
                print("responsible_index_j", responsible_index_j)
                #get the correct index 
                responsible_index = cell_indices_extended_i[responsible_index_j]
                print("responsible_index", responsible_index)
                responsible_indices[i] = responsible_index            
        else:
            print("unknown iou mode")
            sys.exit(1)



        indices = T.arange(0, responsible_indices.shape[0], dtype=T.long)
        
        responsible_indices_1 = T.zeros((converted_box_data.shape[0], ground_truth_boxes.shape[0]), dtype=T.float32, device=self.device)                
        responsible_indices_1[responsible_indices, indices] = 1

        print("responsible_indices_1", responsible_indices_1)

        responsible_indices_any_1 = T.max(responsible_indices_1, dim=1).values
        responsible_indices_any_1 = T.reshape(responsible_indices_any_1, (self.S*self.S*self.B, 1))
        #print("responsible_indices_any_1", responsible_indices_any_1)

        responsible_indices_noobj_1 = T.ones_like(responsible_indices_any_1)
        responsible_indices_noobj_1[responsible_indices_any_1 > 0] = 0
        #print("responsible_indices_noobj_1", responsible_indices_noobj_1)
        
        #responsible_indices_noobj_1 = T.ones((converted_box_data.shape[0], ground_truth_boxes.shape[0]), dtype=T.float32, device=self.device)                
        #responsible_indices_noobj_1[responsible_indices, indices] = 0

        return responsible_indices, responsible_indices_1, responsible_indices_any_1, responsible_indices_noobj_1

    def get_intersected_cells(self, ground_truth_boxes):
        """
        Identifies the cells intersected by each ground truth box.

        :returns intersected_cells_mask: bool tensor with shape (S*S, num ground truth boxes).
        Contains all (cell, ground truth box) pairs.

        :returns intersected_cells_1: float representation of responsible_cells_mask with values 0 and 1.
        Contains all (cell, ground truth box) pairs.

        :returns intersected_cells_1_any_box: float tensor with shape (S*S, 1) with values 0 and 1.
        1 if the cell is intersected by any of the ground truth boxes.
        """
        intersected_cells_mask = T.empty((self.grid_data.shape[0], ground_truth_boxes.shape[0]), dtype=T.bool, device=self.device)
        
        #boolean expression to check whether two AABBs intersect
        #rect_a.left <= rect_b.right && 
        #rect_a.right >= rect_b.left &&
        #rect_a.top >= rect_b.bottom && 
        #rect_a.bottom <= rect_b.top

        #let rect_a be the ground_truth_boxes and rect_b be the grid_data
        #ground_truth_boxes: (x1, y1, x2, y2)
        #grid_data: (grid x index, grid y index, cell min x, cell max x, cell min y, cell max y)
        for i in range(ground_truth_boxes.shape[0]):
            intersected_cells_mask[:,i] = (
                (ground_truth_boxes[i,0] <= self.grid_data[:,3]) & 
                (ground_truth_boxes[i,2] >= self.grid_data[:,2]) &
                (ground_truth_boxes[i,3] >= self.grid_data[:,4]) & 
                (ground_truth_boxes[i,1] <= self.grid_data[:,5]))

        intersected_cells_1 = T.zeros_like(intersected_cells_mask, dtype=T.float32, device=self.device)
        intersected_cells_1[intersected_cells_mask] = 1
        intersected_cells_1_any_box = T.max(intersected_cells_1, dim=1).values
        intersected_cells_1_any_box = T.reshape(intersected_cells_1_any_box, (self.S*self.S, 1))
        return intersected_cells_mask, intersected_cells_1, intersected_cells_1_any_box

    def get_responsible_cells(self, ground_truth_boxes):
        """
        Identifies the cells containing the center of each ground truth box.

        :returns responsible_cells_mask: bool tensor with shape (S*S, num ground truth boxes)

        :returns responsible_cells_1: float representation of responsible_cells_mask with values 0 and 1
        """
        responsible_cells_mask = T.zeros((self.grid_data.shape[0], ground_truth_boxes.shape[0]), dtype=T.bool, device=self.device)

        cell_size = 1 / self.S
        #ground_truth_boxes: (x1, y1, x2, y2)
        #grid_data: (grid x index, grid y index, cell min x, cell max x, cell min y, cell max y)
        responsible_cells_indices = T.zeros(ground_truth_boxes.shape[0], dtype=T.long, device=self.device)
        for i in range(ground_truth_boxes.shape[0]):
            x = (ground_truth_boxes[i,0] + ground_truth_boxes[i,2]) / 2
            y = (ground_truth_boxes[i,1] + ground_truth_boxes[i,3]) / 2
            x_index = int(x / cell_size)
            y_index = int(y / cell_size)
            cell_index = x_index + y_index * self.S
            responsible_cells_mask[cell_index, i] = True
            responsible_cells_indices[i] = cell_index

        responsible_cells_1 = T.zeros_like(responsible_cells_mask, dtype=T.float32, device=self.device)
        responsible_cells_1[responsible_cells_mask] = 1
        return responsible_cells_mask, responsible_cells_1, responsible_cells_indices

    def get_class_probability_map(self, converted_box_data):
        """
        Generates the class probability map containing, for each grid cell, the probability for each class.

        :param converted_box_data: the boxes obtained from prepare_data.

        :returns  class_probability_map: tensor of shape (S*S, C)
        """
        #since predicted classes are identical for each box of a cell, we only need S*S boxes
        #we also need to discard the first 5 values of those boxes, since they are used for
        #coordinates and box confidence
        class_probability_map = converted_box_data[:self.S*self.S, 5:]
        return class_probability_map

    def get_batch_loss(self, forward_result, batch_boxes, batch_labels, lambda_coord = 5, lambda_noobj = 0.5):
        """
        Calculates loss for an entire batch.

        :param forward_result: the tensor obtained from forward. 
        """
        print("get_batch_loss")
        print(forward_result.shape)
        batch_size = forward_result.shape[0]
        total_loss = 0
        for i in range(batch_size):
            converted_box_data = self.prepare_data(i, forward_result)
            ground_truth_boxes = batch_boxes[i]
            ground_truth_label = batch_labels[i]
            loss_i = self.get_loss(converted_box_data, ground_truth_boxes, ground_truth_label, lambda_coord=lambda_coord, lambda_noobj=lambda_noobj)
            total_loss += loss_i
        print("total_loss", total_loss)
        return total_loss

    def get_batch_class_predictions(self, forward_result):
        """
        Get class predictions for an entire batch.

        :param forward_result: the tensor obtained from forward. 
        """
        print("get_batch_class_predictions")
        print(forward_result.shape)
        batch_size = forward_result.shape[0]
        predictions = T.empty((batch_size, self.C))
        for i in range(batch_size):
            print("batch element i", i)
            converted_box_data = self.prepare_data(i, forward_result)
            _, filtered_converted_box_data, _ = self.non_max_suppression(converted_box_data)    
            predictions_i = self.get_class_prediction(filtered_converted_box_data)
            print("predictions_i", predictions_i)
            predictions[i] = predictions_i
        return predictions

    def get_batch_loss_and_class_predictions_and_boxes(self, forward):
        """
        The functions get_batch_loss and get_batch_class_predictions both compute the converted_box_data.
        To save computation time, this method combines those methods

        :param forward_result: the tensor obtained from forward. 
        """
        
    def get_loss(self, converted_box_data, ground_truth_boxes, ground_truth_label, lambda_coord = 5, lambda_noobj = 0.5):
        """
        Calculates loss for a single image.

        :param converted_box_data: the boxes obtained from prepare_data.

        :param ground_truth_boxes: the ground truth boxes.

        :param ground_truth_label: the one hot encoded ground truth label.
        In this task there is only one label for all boxes of an image     
        """
        #if there are no boxes, only penalize confidence of all boxes and class predictions
        if ground_truth_boxes is None:
            converted_box_data_c = converted_box_data[:,4]
            class_probability_map = self.get_class_probability_map(converted_box_data)

            #PART 4: box confidence error (empty cells)
            box_noobj_confidence_errors = (
                T.square(0 - converted_box_data_c)
            )
            print("box_noobj_confidence_errors", box_noobj_confidence_errors)
            part_4 = lambda_noobj * T.sum(box_noobj_confidence_errors)
            print("part_4", part_4)

            #PART 5: classification error
            classification_errors = T.square(ground_truth_label - class_probability_map) 
            print("ground_truth_label", ground_truth_label)  
            print("class_probability_map", class_probability_map)  
            print("classification_errors", classification_errors)  
            part_5 = T.sum(classification_errors)

            return part_4 + part_5

        #extract data
        responsible_indices, responsible_indices_1, responsible_indices_any_1, responsible_indices_noobj_1 = self.get_responsible_indices(converted_box_data, ground_truth_boxes)
        intersected_cells_mask, intersected_cells_1, intersected_cells_1_any_box = self.get_intersected_cells(ground_truth_boxes)
        class_probability_map = self.get_class_probability_map(converted_box_data)
        print("ground_truth_boxes", ground_truth_boxes)
        print("converted_box_data[responsible_indices]", converted_box_data[responsible_indices])
        #ground_truth_boxes: (x1, y1, x2, y2)
        ground_truth_boxes_x1 = ground_truth_boxes[:,0]
        ground_truth_boxes_y1 = ground_truth_boxes[:,1]
        ground_truth_boxes_x2 = ground_truth_boxes[:,2]
        ground_truth_boxes_y2 = ground_truth_boxes[:,3]
        #converted_box_data
        converted_box_data_x1 = converted_box_data[:,0]
        converted_box_data_y1 = converted_box_data[:,1]
        converted_box_data_x2 = converted_box_data[:,2]
        converted_box_data_y2 = converted_box_data[:,3]
        converted_box_data_c = converted_box_data[:,4]
        #reshape for broadcasting
        converted_box_data_x1 = T.reshape(converted_box_data_x1, (*converted_box_data_x1.shape, 1))
        converted_box_data_y1 = T.reshape(converted_box_data_y1, (*converted_box_data_y1.shape, 1))
        converted_box_data_x2 = T.reshape(converted_box_data_x2, (*converted_box_data_x2.shape, 1))
        converted_box_data_y2 = T.reshape(converted_box_data_y2, (*converted_box_data_y2.shape, 1))
        converted_box_data_c = T.reshape(converted_box_data_c, (*converted_box_data_c.shape, 1))

        #calculate loss

        #PART 1: box position error
        #the loss function might be using coordinates relative to the cell
        #we use coordinates relative to the image, resulting in a smaller loss
        #we therefore need to multiply the coordinate differences by the number of boxes (S)
        #but instead of multiplying each difference, it is enough to multiply the result by 
        #lambda_scale = S*S
        #since ((a*x)^2 + (a*y)^2) = a^2*(x^2 + y^2)
        lambda_scale = self.S * self.S
        box_position_errors = responsible_indices_1 * (
            T.square(ground_truth_boxes_x1 - converted_box_data_x1) + 
            T.square(ground_truth_boxes_y1 - converted_box_data_y1)
        )
        print("box_position_errors", box_position_errors)
        part_1 = lambda_coord * lambda_scale * T.sum(box_position_errors)   

        
        #PART 2: box dimension error
        box_dimension_errors = responsible_indices_1 * (
            T.square(
                T.sqrt(ground_truth_boxes_x2-ground_truth_boxes_x1) - 
                T.sqrt(converted_box_data_x2-converted_box_data_x1)
            ) + 
            T.square(
                T.sqrt(ground_truth_boxes_y2-ground_truth_boxes_y1) - 
                T.sqrt(converted_box_data_y2-converted_box_data_y1)
            )
        )
        print("box_dimension_errors", box_dimension_errors)
        part_2 = lambda_coord * T.sum(box_dimension_errors)   


        #PART 3: box confidence error (non empty cells)
        #here we can use responsible_indices_any_1 instead of responsible_indices_1
        #since we do not care about which ground truth box the box is responsible for
        box_confidence_errors = responsible_indices_any_1 * (
            T.square(1 - converted_box_data_c)
        )
        print("box_confidence_errors", box_confidence_errors)
        part_3 = T.sum(box_confidence_errors)   

        
        #PART 4: box confidence error (empty cells)
        box_noobj_confidence_errors = responsible_indices_noobj_1 * (
            T.square(0 - converted_box_data_c)
        )
        print("box_noobj_confidence_errors", box_noobj_confidence_errors)
        part_4 = lambda_noobj * T.sum(box_noobj_confidence_errors)


        #PART 5: classification error
        classification_errors = intersected_cells_1_any_box * T.square(ground_truth_label - class_probability_map) 
        print("classification_errors", classification_errors)  
        part_5 = T.sum(classification_errors)   

        print("part_1", part_1)  
        print("part_2", part_2)  
        print("part_3", part_3)  
        print("part_4", part_4)  
        print("part_5", part_5)  
        print("responsible_indices_1.shape", responsible_indices_1.shape)  
        print("responsible_indices_any_1.shape", responsible_indices_any_1.shape)  
        print("responsible_indices_noobj_1.shape", responsible_indices_noobj_1.shape) 

        #combine parts 
        total_loss = part_1 + part_2 + part_3 + part_4 + part_5
        print("total_loss", total_loss)  
        return total_loss

    def get_class_prediction(self, filtered_converted_box_data):
        data = np.array([
            [0.5, 0.5, 0.6, 0.6, 0.8, 0.1, 0.5, 0.1, 0.1],
            [0.5, 0.5, 0.6, 0.6, 0.9, 0.1, 0.1, 0.5, 0.1],
            [0.7, 0.7, 0.8, 0.8, 0.7, 0.1, 0.5, 0.1, 0.1]
        ])
        filtered_converted_box_data = T.tensor(data, dtype=T.float32)
        
        #special case for no boxes predicted --> no pneumonia
        if filtered_converted_box_data.shape[0] == 0:
            print("no boxes remain")
            no_pneumonia = T.zeros((1, self.C))
            no_pneumonia[0,0] = 1
            return no_pneumonia

        #if at least one box was predicted, calculate class specific confidence
        confidences = filtered_converted_box_data[:, 4]
        confidences = T.reshape(confidences, (*confidences.shape, 1))
        probabilities = filtered_converted_box_data[:, 5:]
        class_specific_confidence = confidences * probabilities
        print("confidences", confidences)
        print("probabilities", probabilities)
        print("class_specific_confidence", class_specific_confidence)

        #there are multiple ways to obtain a single class prediction
        #method 1: use the highest class specific confidence found
        if self.prediction_method == PREDICTION_MAX:
            max_cscs = T.max(class_specific_confidence, dim=0).values
            index = T.argmax(max_cscs)
            #row_indices = T.argmax(class_specific_confidence, dim=0)
            #max_cscs = class_specific_confidence[row_indices]
            #print("row_indices", row_indices)
            print("max_cscs", max_cscs)
            print("index", index)
            prediction = T.zeros((1, self.C))
            prediction[0,index] = 1
            print("prediction", prediction)
            return prediction
        #method 2: use the highest mean class specific confidence found
        if self.prediction_method == PREDICTION_MAX_MEAN:
            mean_cscs = T.mean(class_specific_confidence, dim=0)
            index = T.argmax(mean_cscs)
            print("mean_cscs", mean_cscs)
            print("index", index)
            prediction = T.zeros((1, self.C))
            prediction[0,index] = 1
            print("prediction", prediction)
            return prediction

        print("unknown prediction method", self.prediction_method.shape)
        sys.exit()