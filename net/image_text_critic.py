import torch 
import torch.nn as nn 
import torch.nn.functional  as F 
from transformers import CLIPImageProcessor,CLIPForImageClassification,CLIPVisionModel
from net.UNET  import UNetDenoise
from torch.distributions import  Normal
class criticNet(nn.Module):
    def __init__(self,config1, config2,word_sequence,image_sequence):
     super(criticNet,self).__init__()
     self.config1=config1
     self.config2=config2 
     self.kernel_size=config1['image_tokenizer_kernel_size'][image_sequence]
     self.stride=config2['parameters']['stride']
     self.padding=config2['parameters']['padding']
     self.embed_dim=config1['encoder-decoder_embed_dim'][word_sequence]
     self.image_dim=config1['image_tokenizer_embed_dim'][image_sequence]
     self.in_channels=config1['in_channels']
     self.out_channels=config1['out_channels']
     self.n_heads=config2['parameters']['num_heads']
     self.linear=nn.Linear(in_features=self.image_dim , out_features=self.embed_dim)
     self.conv2d=nn.Conv1d(in_channels=self.in_channels,out_channels=self.out_channels,kernel_size=self.kernel_size,padding=self.padding,stride=self.stride)
     self.unet_depth=config2['unet_depth']
     self.cross_depth=config2['parameters']['num_attention_layers']
     self.cross_attention =nn.ModuleList([nn.MultiheadAttention(embed_dim=self.embed_dim,num_heads=self.n_heads)for _ in range(self.cross_depth)])
     self.batchnorm =nn.BatchNorm1d(num_features=1)
     self.batch_size=config2['parameters']['batch-size']
    
     self.kernel_height=(self.frame_height-self.kernel_size+2*self.padding)//self.stride +1
     self.kernel_width=(self.frame_width-self.kernel_size+2*self.padding)//self.stride+1
     self.block_size=self.kernel_height*self.kernel_width+1 # 1  refers to tokenizer IDs
     self.unet=nn.ModuleList( [UNetDenoise(n_channels=self.block_size,n_classes=1) for i in range(self.unet_depth)])
     self.relu=nn.ReLU()
     self.num_conv=config2['num_attention_layers']
     self.conv2d=nn.ModuleList([nn.ConvTranspose2d(in_channels=self.block_size,out_channels=1,kernel_size=self.kernel_size,padding=self.padding,stride=self.stride) for  _ in range(self.num_conv )] )
    
     self.linear2=nn.Linear(in_features=self.block_size, out_features=1)
    
    def forward(self,image,description): #image -> [batch_size,block_size ,image_dim]
        if description.dim()==1:
            description=description.unsqueeze(0).expand(self.block_size,-1)
        description=description.unsqueeze(0).expand(self.batch_size,-1,-1)
        image_features=image[:,:-1,:]
        image_word =self.linear(image_features)
        for attn in self.cross_attention:
            output=attn(description,image_word,image_word)[0]
            description=output
        
        output=self.linear2(output.transpose(0,2,1).squeeze(2))
       
        
        output_mean=torch.mean(output,dim=0)
        output_std=torch.std(output,dim=0)
        
        return output_mean ,output_std,torch.ceil_(self.relu(output))