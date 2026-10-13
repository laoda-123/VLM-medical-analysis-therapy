import numpy  as np 
import torch
import torch.nn as nn
import torch.nn.functional  as F
import os
import torchvision.transforms as transforms
from PIL import Image
import clip
from  envs.env_builder import load_env_file
from net.UNET import UNetDenoise
from abc import ABC,abstractmethod
import torchvision.transforms as transforms
from  flask import Flask,app,request,jsonify,render_template,Response
import random
number=torch.random.seed()





def encoder(texts):
    
    device="cuda" if torch.cuda.is_available() else "cpu"
    model,preprocess=clip.load("ViT-B/32",device=device)
    texts=clip.tokenize(texts).to(device)
    with torch.no_grad():
      text_features=model.encode_text(texts)


    # 获取Token嵌入
    return text_features
 
class resnet_layers(nn.Module):
   def __init__(self,input_channel,dict,kernel_size,expansion=2.0):
    super().__init__()
    self.embed_dim =dict["ch1"]
    self.input_channel=input_channel
    self.kernel_size=kernel_size
    self.conv2 =nn.Conv2d(input_channel,self.embed_dim,kernel_size=kernel_size,padding=0)
    self.bn1=nn.BatchNorm2d(self.embed_dim)
    self.conv3=nn.Conv2d(self.embed_dim,dict["ch2"],kernel_size=kernel_size,padding=0)
    self.bn2=nn.BatchNorm2d(dict["ch2"])
    self.conv4=nn.Conv2d(dict["ch2"],dict["ch2"]*expansion,kernel_size,padding=0)  #ouput ->(output_channel,)
   def  forward(self,x):
    con_1=self.bn1(self.conv2(x))
    out=F.relu(con_1)
    con_2=self.bn2(self.conv3(out))
    out=F.relu(con_2)
    out =self.conv4(out)
    out=F.relu(out)
    return out

def get_size(x):
   return x.shape[0],x.shape[1],x.shape[2]
def generate(x):
    batch_size,patch_size,embed_dim=get_size(x)
    return torch.randn(size=(batch_size,patch_size,embed_dim))


class stable_diffusion(nn.Module):
 def __init__(self,config1,config2,sequence,device): #config1 is image_description.yaml and config2 is text_image.yaml
   super().__init__()
   self.device=device
   self.embed_dim=config1['Bert_embed_dim'][sequence]
   self.unet_depth=config2['unet_depth']
   self.num_cross=config2['cross_attention_layers']
   self.in_channels=config2['parameters']['in_channels']
   self.out_channels=config2['parameters']['out_channels']
   
   self.kernel_size=config2['parameters']['kernel_size']
   self.stride=config1['parameters']['stride']
   self.padding=config1['parameters']['padding']
   self.num_heads=config2['parameters']['num_heads']
   self.num_cross=config2['cross_attention_layers']
   self.frame_height=config2['frame_height']
   self.frame_width=config2['frame_width']
   self.block_height=(self.frame_height-self.kernel_size+2*self.padding)//self.stride +1
   self.block_width=(self.frame_width-self.kernel_size+2*self.padding)//self.stride+1
   self.image_blocks=self.block_height*self.block_width
   
   self.linear=nn.Linear(in_features=self.embed_dim,out_features=self.image_blocks)
   self.cross_attention_layers=nn.ModuleList([nn.MultiheadAttention(embed_dim=self.image_blocks,num_heads=self.num_heads) for _ in range(self.num_cross)])
   self.batch_size =config2['parameters']['batch_size']
   
   self.transpose2D=nn.ConvTranspose2d(in_channels=self.in_channels,out_channels=self.out_channels ,kernel_size=self.kernel_size,stride=self.stride  ,padding=self.padding)
   self.unet=nn.ModuleList([UNetDenoise(n_channels=self.out_channels,n_classes=4)for _ in range(self.unet_depth)])
   self.cross_attention=nn.ModuleList([nn.MultiheadAttention(embed_dim =self.embed_dim,num_heads=self.num_heads) for _ in range(self.num_cross) ])
   self.batchnorm=nn.BatchNorm2d(num_features=self.out_channels)
   
   self.noise=nn.Parameter(torch.randn(self.in_channels,self.embed_dim))
   self.relu=nn.GELU()
 def forward(self,x,norm_first=False ):
   if x.dim()==1:
    x=x.unsqueeze(0)
   x=x.unsqueeze(1)
   noise=self.noise.unsqueeze(0).expand(self.batch_size,-1,-1).to(self.device)
   if norm_first==False:
    for net  in self.cross_attention:
     blur_channels=net(noise,x,x)[0]
     noise=self.batchnorm(blur_channels)
     
   else:
    for exp_net in self.cross_attention:
     query_input=self.batchnorm(noise)+noise
     blur_channels=exp_net(query_input,x,x)[0]
     query_input=blur_channels
   image_features=self.linear(blur_channels).view(self.batch_size,self.in_channels,self.block_height ,self.block_width)
   noise_image=self.relu(self.transpose2D(image_features))
   
   for net in self.unet:
       clear_img=net(noise_image)
       noise_image=clear_img
  
   return self.relu(self.noise_image)

def tensor_to_pil_image(tensor):
    # 1. 移除批次维度（如果有）
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # 2. 将张量值调整到[0, 1]范围
    tensor = tensor.clamp(0, 1)
    
    # 3. 使用ToPILImage转换为PIL图像
    to_pil = transforms.ToPILImage()
    pil_image = to_pil(tensor)
    
    return pil_image
def calculate_distance(object_size, focal_length, pixel_size, pixel_count):
    """
    calculate the distance  between object and the len
    :param object_size: 物体实际大小(unit:m)
    :param focal_length: 镜头焦距(unit:m)
    :param pixel_size: 像素物理尺寸 (unit: m/pixel)
    :param pixel_count: 物体在图像中的像素大小(unit:像素)
    :return: 镜头到物体的距离 (unit:m)
    """
    if pixel_count == 0:
        return 0
    distance = (object_size * focal_length) / (pixel_count * pixel_size)
    return distance
def main():
 
 return 
  
 
if '__name__'== '__main__':
  main()