from basic_network import *

class SimpleCNN(BasicNetwork):
    """
    Neural Network test
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()        
        print("__init__ SimpleCNN")
        self.leaky_slope = 0.1  
        #setup layers
        self.conv_1 = nn.Conv2d(
            in_channels=3, 
            out_channels=64, 
            kernel_size=7, 
            stride=2, 
            padding=3)

        size_after_conv_1 = self.get_conv_out_size(
            input_size=448, 
            kernel_size=7,            
            stride=2,
            padding=3)

        flattened = 64*size_after_conv_1*size_after_conv_1

        print("size_after_conv_1", size_after_conv_1)
        print("flattened", flattened)

        self.full_1 = nn.Linear(flattened, 100)
        self.full_2 = nn.Linear(100, 10)

    def forward(self, x):     
        #print("forward:", x.size())
        x = F.leaky_relu(self.conv_1(x), negative_slope=self.leaky_slope)
        self.print_debug("conv_1", x.size())
        #tensor needs to be flattened to 2d to work with fully connected layer
        #first dimension for batch, second for the flattened part
        x = T.flatten(x, start_dim=1)
        #print("D", x.size())
        x = F.relu(self.full_1(x))
        #print("F", x.size())
        x = self.full_2(x)
        #print(x.size())
        return x