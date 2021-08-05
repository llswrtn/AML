from basic_network import *

class Yolo(BasicNetwork):
    """
    Neural Network test
    """
    def __init__(self, number_of_classes, dropout_p=0.5):
        super(Yolo, self).__init__()        
        print("init Yolo")
        self.leaky_slope = 0.1  
        self.number_of_classes = number_of_classes
        self.dropout_p = dropout_p
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
        S = 7
        B = 2
        C = number_of_classes
        self.values_per_cell = B*5+C
        self.layer_30_full = nn.Linear(4069, S*S*(B*5+C))
        #endregion

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
        x = T.reshape(x, (x.shape[0], 7, 7, self.values_per_cell))
        return x

        