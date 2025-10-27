import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from collections import OrderedDict

# Depthwise Convolution Module
class DepthWise_Conv(nn.Module):
    def __init__(self, in_fts, stride=(1, 1))->None:
        super(DepthWise_Conv, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_fts,in_fts,kernel_size=3,stride=stride,padding=1, groups=in_fts,bias=False),
            nn.BatchNorm2d(in_fts),
            nn.ReLU6(inplace=True)
        )
    def forward(self,x):
        return self.conv(x)
    
# Pointwise Convolution Module
class PointWise_Conv(nn.Module):
    def __init__(self,in_channels,out_channels)->None:
        super(PointWise_Conv,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(out_channels)
            
        )
        
    def forward(self,x):
        return self.conv(x)

#Bottleneck Layer when Stride =1
class NetForStrideOne(nn.Module):
    def __init__(self,in_channels,out_channels,expandsion)->None:
        super(NetForStrideOne,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels,expandsion*in_channels,kernel_size=1,bias=False),
            nn.BatchNorm2d(expandsion*in_channels),
            nn.ReLU6(inplace=True)
        )
        
        self.dw=DepthWise_Conv(expandsion*in_channels)
        self.pw=PointWise_Conv(expandsion*in_channels,out_channels)
        
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.expandsion=expandsion
        
    def forward(self,x):
        if self.expandsion==1:
            out=self.dw(x)
            out=self.pw(out)
        else:
            out=self.conv1(x)
            out=self.dw(out)
            out=self.pw(out)
            
        if self.in_channels==self.out_channels:
            return x+out
        
        return out
    
#Bottleneck Layer when stride=2

class NetForStrideTwo(nn.Module):
    def __init__(self,in_channels,out_channels,expandsion)->None:
        super(NetForStrideTwo,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels,expandsion*in_channels,kernel_size=1,bias=False),
            nn.BatchNorm2d(expandsion*in_channels),
            nn.ReLU6(inplace=True)
        )
        self.dw=DepthWise_Conv(expandsion*in_channels,stride=2)
        self.pw=PointWise_Conv(expandsion*in_channels,out_channels)
        self.expandsion=expandsion
        
    def forward(self,x):
        if self.expandsion==1:
            out=self.dw(x)
            out=self.pw(out)
            
        else:
            out=self.conv1(x)
            out=self.dw(out)
            out=self.pw(out)
            
        return out
    
#MobileNetV2 Architecture
class MobileNetV2(nn.Module):
    def __init__(self,bottleneckLayerDetails,in_channels,num_classes=1000,width_mult=1.0)->None:
        super(MobileNetV2,self).__init__()
        self.bottleneckLayerDetails=bottleneckLayerDetails
        
        self.num_classes=num_classes
        self.width_mult=width_mult
        
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels,round(width_mult*32),kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(round(width_mult*32)),
            nn.ReLU6(inplace=True)
        )
        
        self.in_channels=round(width_mult*32)
        
        #define bottleneck layers 
        self.layersConstructed=self.constructLayer()
        
        #Top layers after bottleneck
        self.feature=nn.Sequential(
            nn.Conv2d(self.in_channels,round(width_mult*1280),kernel_size=1,bias=False),
            nn.BatchNorm2d(round(width_mult*1280)),
            nn.ReLU6(inplace=True),
            
        )
        
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.outputLayer=nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(round(width_mult*1280),self.num_classes,kernel_size=1)
        )
        
    def forward(self,x):
        x=self.conv1(x)
        x=self.layersConstructed(x)
        x=self.feature(x)
        x=self.avgpool(x)
        x=self.outputLayer(x)
        return x
    
    def constructLayer(self):
        itemIndex=0
        block=OrderedDict()
        #interating the defined layers details
        for lItem in self.bottleneckLayerDetails:
            t,out,n,stride=lItem
            #if width multipler is mentioned then perform this line
            out_channels=round(self.width_mult*out)
            #for stride=1
            if stride==1:
                #constructed the NetForStrideOne layer n times
                for nItem in range(n):
                    block[str(itemIndex)+"_"+str(nItem)]=NetForStrideOne(self.in_channels,out_channels,t)
                    self.in_channels=out_channels
            
            #for stride =2
            
            elif stride==2:
                #first layer constructed with NetForStrideTwo module once only
                block[str(itemIndex)+"_"+str(0)]=NetForStrideTwo(self.in_channels,out_channels,t)
                self.in_channels=out_channels
                #remaining n-1 layers constructed with NetForStrideOne module
                for nItem in range(1,n):
                    block[str(itemIndex)+"_"+str(nItem)]=NetForStrideOne(self.in_channels,out_channels,t)
                    
            itemIndex +=1
        return nn.Sequential(block)
    
if __name__=="__main__":
    #Bottleneck layer details as per MobileNetV2 architecture
    bottleneckLayerDetails=[
        #t,out,n,stride
        [1,16,1,1],
        [6,24,2,2],
        [6,32,3,2],
        [6,64,4,2],
        [6,96,3,1],
        [6,160,3,2],
        [6,320,1,1]
    ]
    
    model=MobileNetV2(bottleneckLayerDetails,in_channels=3,num_classes=1000,width_mult=1.0)
    print(model)
    
    #print the model summary
    summary(model,input_size=(1,3,224,224),device="cuda")