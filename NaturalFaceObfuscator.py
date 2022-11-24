import numpy as np
import torch
from torchvision import datasets, models, transforms
from PIL import Image
import os
import glob
from torch.autograd import Variable
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.utils import save_image

model = torch.load("newmodel101.pt", map_location=torch.device('cpu')) #load your obfuscation detector
trans =  transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

resnet = models.resnet18(pretrained=True) #load your feature extractor 

featureExtractor = create_feature_extractor(resnet, return_nodes=['avgpool'])

tImgStr = "53864_PubFig.png" #set to whatever image you want
tImg = Image.open(tImgStr)
tImg = tImg.resize((224,224))
tIm = trans(tImg)
tIm = tIm.view(1,3,224,224)
tIm.requires_grad = True


def nu(mImg):
  fakeness = torch.maximum(model(mImg)[0,0] + torch.tensor(2.5), torch.tensor(0.0))
  return fakeness


def rho(mImg):
  targetFeatures = featureExtractor(tIm)['avgpool'].squeeze(0).squeeze(1).squeeze(1)
  modFeatures = featureExtractor(mImg)['avgpool'].squeeze(0).squeeze(1).squeeze(1)
  df = torch.sqrt(torch.sum(torch.square(targetFeatures - modFeatures)))
  return df 


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        with torch.no_grad():
          for t, m, s in zip(tensor, self.mean, self.std):
              t.mul_(s).add_(m)
              # The normalize code -> t.sub_(m).div_(s)
          return tensor


ITR = 200

for file in glob.glob("CroppedImages/*.png"): #have images you want to obscure in this folder, or change folder name
    if ("cloaked" not in file):
        print(file)
        im = Image.open(file)
        im = im.resize((224,224))   # Fawkes img size
        if (np.array(im).shape == (224,224,3)):
            source = trans(im)
            source = source.view(1,3,224,224)

        source.requires_grad=True

        modifier = Variable(torch.randn  (   (1,3,224,224)), requires_grad=True) 
        for i in range(ITR):
            modified = source + modifier * 1e-1
            
            M_s = rho(modified)
            M_o = nu(modified)
            loss =  M_s + M_o
            loss.backward()

            modifier = torch.Tensor.detach(modifier - modifier.grad * 20) 
            modifier.requires_grad = True 
            print("{}/{} - Total loss: {} Obfuscated: {} Similarity: {}".format(i, ITR , loss.item(), M_o.item(), M_s.item()))  

        unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        modified = modified.squeeze(0)
        save_image(unorm(modified), "{}_natural_cloaked.png".format(ct))


