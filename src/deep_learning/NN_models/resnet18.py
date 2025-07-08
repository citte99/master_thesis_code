import torch
import torch.nn as nn
from deep_learning.registry import MODEL_REGISTRY


class BasicBlock(nn.Module):
    expansion = 4
    #tipically, the downsample layer is set to have output channels of out_channels*expansion, where out_channels is the out_channels of the convolutional layer
    #usually, the expansion is more tipical of a bottleneck layer.

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        '''
        In the initialization, we set all the layers to have appropriate geometry. We are calling the init functions of the layers.
        '''

        #I guess downsample reduces the size of the output of each filter. Question: isn't that already set by stride?
        #The downsample parameters is a function of both the stride and number of channels


        super(BasicBlock, self).__init__()
        self.conv1 =nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        #questions: the number of channels out sets the number of filters, correct? yes
        #question: all the filters have the same size: 3x3, correct? yes
        #question: why is the bias not set here? The bias is set by the batchnorm, so it would be redundant

        self.bn1= nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        self.conv2= nn.Conv2d (out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 =nn.BatchNorm2d(out_channels)

        #optional downsample layer, provided already implemented
        self.downsample= downsample
        self.stride= stride #here we set stride as a paramter of the block. This means we will be able to access it. Stride gives the reduction of size of each 2d featuremap

    def forward(self, x):
        identity = x #for the skip connection. 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out =self.bn2 (out)


        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        out = self.relu (out)

        return out



class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(ResNet, self).__init__()
        '''
        I guess this part could be trimmed a lot
        '''
        self.in_channels = 64 #starting number of channels
        
        #Perform an intial convolution: this is presuming it gets 1 channel.
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64) 
        self.relu =nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d (kernel_size=3, stride=2, padding=1)


        # Create four layers (groups) of residual blocks.
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Global average pooling: each channel is reduced to one value.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Fully connected layer to produce the final class scores.
        self.fc = nn.Linear(512 * block.expansion, num_classes)



    def _make_layer(self, block, out_channels, blocks, stride=1):
        '''
        we are linking many residual blocks 

        be aware: expansion is tipical of bottleneck layers. For this reason, its
        implemented in a general downsample layer, but will be used just by bottleneck layers.
        '''

        downsample = None
        
        if stride!=1 or self.in_channels != out_channels*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels*block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                #comment: this downsample is applied at each simpleblock layer, so clearly it shares its stride

                nn.BatchNorm2d(out_channels*block.expansion),
            )


        layers=[]
        layers.append(block(
            self.in_channels, out_channels, stride, downsample
        ))
        self.in_channels=out_channels*block.expansion
        #who performed the expansion? the downsample did it for for the shortcut,
        for _ in range (1, blocks):
            #be aware, the following blocks all have stride 1!
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Process the input through the initial layers.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Pass through the four residual layers.
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling.
        x = self.avgpool(x)
        # Flatten the output (except the batch dimension).
        x = torch.flatten(x, 1)
        # Final fully connected layer.
        x = self.fc(x)
        return x

    
    
@MODEL_REGISTRY.register()
def ResNet18(num_classes=2):
    # ResNet-18 configuration: [2, 2, 2, 2] blocks in each layer.
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

# Example usage:
if __name__ == "__main__":
    model = ResNet18(num_classes=2)
    print(model)
    
    # Create a dummy input: batch of 8 images, 3 channels, size 224x224.
    x = torch.randn(8, 1, 224, 224)
    logits = model(x)
    print("Output shape:", logits.shape)  # Expected: (8, 10)
