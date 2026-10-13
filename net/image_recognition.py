import numpy as np 
import torch 
import torch.nn as nn 
from  PIL import Image
import os
from transformers import Blip2Model,Blip2Processor,Blip2PreTrainedModel,Blip2ForConditionalGeneration
import torch.nn.functional as F
from datetime import datetime
import pytz
from sentence_transformers.util import cos_sim
from envs.env_builder import  load_env_file
import torchvision.transforms as transforms
import torchvision.models as models
from envs.env_run import run_sentence_transformer

from sklearn.metrics.pairwise import cosine_similarity
time=pytz.timezone("Asia/Shanghai")

env_config1=load_env_file("J:/image_language_model/config/image_description.yaml")
env_config2=load_env_file("J:/image_language_model/model_config.yaml")
env_config3=load_env_file("J:/image_language_model/image-text.yaml")



class PatchEmbedding(nn.Module):
    """Patch嵌入层"""
    def __init__(self,config1,config2,sequence1,sequence2):
        super().__init__()
        device="cuda" if torch.cuda.is_available() else "cpu"
        self.embed_dim=config2['Bert_embed_dim'][sequence1]
        self.kernel_size=config2["parameters"]["kernel_size"]
        self.stride=config1["parameters"]["stride"]
        self.padding=config1["parameters"]["padding"]
        self.height=config2["embed_dim"]["image"][0]
        self.width=config2["embed_dim"]["image"][0]
        self.block_size=((self.height+2*self.padding-self.kernel_size)/self.stride+1)*((self.width+2*self.padding-self.kernel_size)/self.stride+1)
        self.linear=nn.Linear(self.block_size,self.embed_dim)
        self.kernel_size=config2['']
        
        self.projection=nn.Conv2d(in_channels=3,out_channels=1,kernel_size=self.kernel_size,stride=self.stride,padding=self.padding)
        
       

    def forward(self, x):
        """
        Args:
            x: 输入图片的张量,形状为(batch_size,inputs_ids,channel,height,width)
            
        Returns:
            out: 输出图片的投影序列,形状为 (batch_size, num_patches, embed_dim) 3个通道的
        """
        if x.dim() ==5:
          x=x.squeeze(1)
        inputs=self.projection(x)
        inputs=inputs.squeeze(1)
        inputs=inputs.flatten(1)
        results=self.linear(inputs)
        return results


#channel cross attention
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio),
            nn.ReLU(),
            nn.Linear(in_planes // ratio, in_planes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
# our group  use space cross attention

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=14):
        super().__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=kernel_size,stride=kernel_size, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.transpose2d=nn.ConvTranspose2d(in_channels=3,out_channels=1,kernel_size=kernel_size,stride=kernel_size,padding=0)
    def forward(self, x):
        # 计算空间注意力权重
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv(attn)
        attn = self.sigmoid(attn)
        attm=x*attn
        original_img=self.transpose2d(attm)
        return original_img
    

class PositionwiseFeedForward(nn.Module):
    """位置前馈网络"""
    def __init__(self, config1):
        super().__init__()
        embed_dim=config1["embed_dim"]["sentence"]
        self.layerNorm=nn.LayerNorm(embed_dim)
        hidden_dim=config1["hidden_dim"]
        dropout=config1["dropout"]
        self.forward_layer=nn.Sequential(
          nn.Linear(embed_dim, hidden_dim),
          nn.Linear(hidden_dim, embed_dim),
          nn.Dropout(p=dropout),
          nn.GELU())
        self.norm_first=config1["norm_first"]
    def forward(self, x):
        """
        Args:
            x: 输入序列，形状为 (batch_size, seq_len, embed_dim)
            
        Returns:
            out: 输出序列，形状为 (batch_size, seq_len, embed_dim)
        """
        if self.norm_first==True:
         x=x+self.layerNorm(x)
         output=self.forward_layer(x)
        else:
         output=self.layerNorm(x+self.forward_layer(x))
        return output
    
class LastLayer(nn.Module):
    def __init__(self, config1):
       super().__init__()
       self.softmax=nn.Softmax()
       self.config1=config1
    def forward(self,x):
       return self.softmax(x)

class image_transform(nn.Module) :
    def __init__(self,config1,config2,sequence): # config1 refers to the image description while config2 refers to image text.yaml
       super(image_transform,self).__init__()
       
       self.embed_dim=config1['encoder-decoder_embed_dim'][sequence ]
       self.image_dim=config1['image_tokenizer_embed_dim'][self.img_seq]
       self.kernel_size=config2['parameters']['kernel_size']
       self.in_channels=config1['in_channels']
       self.out_channels=config1['out_channels']
       self.padding=config2['parameters']['padding']
       self.stride=config2['parameters']['stride']
       self.frame_height=config2['frame_width']
       self.frame_weight=config2['frame_height']
       self.batch_size=config2['batch-size']
       self.nhead=config2['parameters']['num_heads']
       self.encoder=nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=self.embed_dim,nhead=self.nhead),num_layers=6)
       self.linear=nn.Linear(in_features=self.image_dim, out_features=self.embed_dim)      
       self.num_attention=config2['parameters']['num_attention_layers']
       self.cross_attention=nn.ModuleList([nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.nhead) for _ in range(self.num_attention)])
       
       self.norm2d=nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,kernel_size=self.kernel_size,stride=self.stride)
       self.kernel_height=(self.frame_height-self.kernel_size+2*self.padding)//self.stride +1
       self.kernel_width=(self.frame_width-self.kernel_size+2*self.padding)//self.stride+1
       self.block_size=self.kernel_height*self.kernel_width+1 # 1  refers to tokenizer IDs
       self.batchnorm=nn.BatchNorm1d(num_features=self.block_size)
       
    def forward(self, x,prompts,norm_first=False): #x shape[batch_size,patch_size,image_dim  ,prompts(maybe one or more sentences),all input are tokenized features, x is torchvision format(0~1)
        if prompts.dim()==1:
            prompts.unsqueeze(0)
        prompts.unsqueeze(0).expand(self.batch_size,-1, -1)
        image_text_feature=self.linear(x)
        if norm_first == False:
            input=self.batchnorm(image_text_feature)
            for net  in self.cross_attention:
              output=net(prompts,input,input)[0]
              input=output
        else:
            for net in self.cross_attention:
              output=net(prompts,image_text_feature,image_text_feature)[0]
              image_text_features=self.batchnorm(output+prompts)
        
        ids=output[:,-1,:]
        output_ids=ids.squeeze(1)
        
        return  torch.ceil_(output_ids)
    
def cal_sim(in1,in2): 
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    in1=transform(in1).unsqueeze(0)
    in2=transform(in2).unsqueeze(0)
    image1=extract_feature(in1)
    image2=extract_feature(in2)
    score=cosine_similarity(image1,image2)
    return score

def batch_similarity(env_path,sentence1,sentence2):
    model_encode=run_sentence_transformer(env_path,2)
    vector1=model_encode.encode(sentence1)
    vector2=model_encode.encode(sentence2)
    return cos_sim(vector1,vector2)
def extract_feature(ten):
    model = models.resnet50(pretrained=True)
    model.eval()
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    with torch.no_grad():
        features = feature_extractor(ten)
    features = features.view(features.size(0), -1).numpy()
    return features
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xxl",
    device_map="auto",  # 自动分配GPU，若无GPU则用CPU
    torch_dtype=torch.float16
)
def generate_caption(image_path,image_text):
    # 加载图片
    # 预处理
    image=Image.open(os.path.join(image_path)).convert("RGB")
    inputs = processor(images=image, text=image_text)
    # 生成描述
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    # 解码输出
    caption = processor.decode(generated_ids[0], skip_special_tokens=True)
    return caption
class SARSA():
   def __init__(self,stride,embed_dim,num_heads,kernal_size):
      super( ).__init__()
      self.stride=stride
      self.embed_dim=self.embed_dim
      self.kernal_size=kernal_size 
      self.num_heads=num_heads
def main():
    start_time=datetime.now(time).strftime("%Y-%m-%d %H:%M:%S")
    

    end_time=datetime.now(time).strftime("%Y-%m-%d %H:%M:%S")
    
