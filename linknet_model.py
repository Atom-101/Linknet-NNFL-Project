import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DecoderBlock(nn.Module):
    def __init__(self,m,n,up=True):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(m,m//4,1,bias=False),
                                   nn.BatchNorm2d(m//4),
                                   nn.ReLU(inplace=True))
        self.fc = nn.Sequential(nn.ConvTranspose2d(m//4,m//4,3,stride=2 if up else 1, padding=1, output_padding=1 if up else 0),
                                nn.BatchNorm2d(m//4),
                                nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(m//4,n,1,bias=False),
                                   nn.BatchNorm2d(n),
                                   nn.ReLU(inplace=True))                                
    def forward(self,x):
        return self.conv2(self.fc(self.conv1(x))) 

class Linknet(nn.Module):
    def __init__(self,classes,pre=True):
        super().__init__()
        '''self.head = nn.Sequential(nn.Conv2d(3,64,7,stride=2,padding=3),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True),
                                  nn.Maxpool2d(3,stride=2))        
        self.encoders = []
        self.encoders.append(EncoderBlock(64,64,up=False))
        for i in range(3):
            self.encoders.append(EncoderBlock(64*2**i,64*2**(i+1))
        self.encoders = nn.Sequential(*self.encoders)'''    
        self.encoder = nn.Sequential(*list(models.resnet18(pre).children())[:-2])
        self.decoders = []
        self.decoders.append(DecoderBlock(64,64,up=False))
        for i in range(3):
            self.decoders.append(DecoderBlock(64*2**(i+1),64*2**i))
        self.decoders = nn.Sequential(*self.decoders)
        self.out = nn.Sequential(nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1),
                                 nn.BatchNorm2d(32),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(32,32,3,stride=1,padding=1),
                                 nn.BatchNorm2d(32),
                                 nn.ReLU(inplace=True),
                                 nn.ConvTranspose2d(32,classes,3,stride=2,padding=1,output_padding=1))
    def forward(self,x):
        x = self.encoder[:4](x)
        activs = [x]
        inp = x
        for i in [4,5,6,7]:
            inp = self.encoder[i](inp)
            activs.append(inp)
        out = activs.pop(-1)
        for i in reversed(range(4)):
            try:
                out = self.decoders[i](out) + activs[i]
            except:
                out = self.decoders[i](out)
                _,_,h,w = activs[i].shape
                out = F.interpolate(out,size=(h,w))
                out += activs[i]
        return self.out(out)
                                 
                                 
if __name__ == '__main__':
    # testing
    m = Linknet(1,False)
    assert m(torch.randn(1,3,64,64)).shape == torch.Size([1,1,64,64])
        
        
        
