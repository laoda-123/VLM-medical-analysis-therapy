import torch.nn as nn 
import torch 
from envs.env_builder import load_env_file
from UNET import UNetDenoise
import torchvision.transforms as transforms
from envs.env_run import run_sentence_transformer
class StepNet(nn.Module):
    def __init__(self,config1,config2,sequence1,device): #config1 is image_description.yaml while config2 is text_image.yaml 
        super(StepNet, self).__init__()
        self.config1=config1
        self.config2=config2
        self.unet_depth =config2['unet_depth']
        self.cross_depth=config2['cross_attention_layers']
        
        self.embed_dim=config1['Bert_embed_dim'][sequence1]
        self.nhead=config2['parameters']['num_heads']
        self.frame_height=config2['frame_height']
        self.frame_width=config2['frame_width']
        
        self.stride=config2['parameters']['stride']
        self.kernel_size=config2['parameters']['kernel_size']
        self.padding=config2['parameters']['padding']
        self.block_height=(self.frame_height-self.kernel_size+2*self.padding)//self.stride+1
        self.block_width=(self.frame_width-self.kernel_size+2*self.padding)//self.stride+1
        self.block_size=self.block_width*self.block_height
        self.in_channels=config2['parameters']['in_channels']
        self.out_channels=config2['parameters']['out_channels']
        self.conv2d=nn.Conv2d(in_channels=self.out_channels,out_channels=self.in_channels,kernel_size =self.kernel_size,stride=self.stride,padding=self.padding)
        self.linear1=nn.Linear(self.block_size,self.embed_dim)
        self.linear2=nn.Linear(in_features=self.embed_dim,out_features=self.block_size)
        self.image_encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.embed_dim,nhead=self.nhead),num_layers=self.cross_depth)
        self.cross_attention=nn.ModuleList([nn.MultiheadAttention(embed_dim=self.embed_dim,num_heads=self.nhead) for _ in range(self.cross_depth)])
        self.conv2d=nn.ConvTranspose2d(in_channels=4,out_channels=4, kernel_size =self.kernel_size, padding=self.padding, stride =self.stride)
        self.unet =nn.ModuleList([UNetDenoise(n_channels=self.in_channels,n_classes =self.out_channels, bilinear=False) for _ in range(self.unet_depth)]).to(device)
        self.relu=nn.GELU() 
        self.batch_size=config2['parameters']['batch_size']
        self.memory=nn.LSTM()
    def forward(self,x,image,norm_first=False): #x refers  to previous output image frame,x1 refers to the  present word clues(inferring pattern)
        if x.dim()==1: 
            x=x.unsqueeze(0)
        img_channels=self.conv2d(image).squeeze(2)
        word_img_proj=self.linear1(img_channels)
        word_channels=x.unsqueeze(1).expand(self.batch_size,-1,-1)
        if norm_first==False:
         for net in self.cross_attention:
            img_word_mix=net(word_channels,img_channels,img_channels)[0]
            word_channels=self.relu(img_word_mix)
        else:
         for net in self.cross_attention:
            word_query=self.relu(word_img_proj)+word_img_proj
            img_word_mix=net(word_query, img_channels,img_channels)[0]
        blur_frame=self.relu(img_word_mix)
        for net in self.unet:
           blur_frame=net(blur_frame)
          
        return self.relu(blur_frame)