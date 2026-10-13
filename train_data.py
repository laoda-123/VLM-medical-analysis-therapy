import numpy as np 
from torch.utils.data import Dataset,DataLoader
import torch
import json
from PIL import Image
import os

import torch.nn as nn
import argparse
import matplotlib.pyplot  as plt
from envs.env_builder import load_env_file
import tqdm
from net.image_recognition import image_transform
from net.image_text_critic import criticNet
from net.Emboided_SAC import environment_model
from envs.env_run import  run_sentence_transformer
import torchvision.transforms as transforms



rng=np.random.default_rng()
#use for classification of image 
import os 

# labelling
class ImageLoader(Dataset):
    def __init__(self, npz_path, transform):
        self.npz_path=npz_path
        self.transform = transform
        data=np.load(npz_path, allow_pickle=True) 
        self.indication=data['indication']
        self.description=data['description']
        self.image_path=data['image_path']
    
    def __len__(self):
        return len(self.indication)

    def __getitem__(self, idx):
        image_file = self.image_path[idx]
        
        image = Image.open(image_file).convert('RGBA')
        indication,description=self.indication[idx],self.description[idx]
        
        if self.transform:
            image = self.transform(image)

        return image, indication,description

    
def train(args):
    device =args.device 
    npz_path=args.npz_path
    config1=args.config1
    config2=args.config2 
    sequence=args.sequence
    img_sequence=args.img_sequence
    batch_size=args.batch_size
    image_normalize=transforms.Compose([
        transforms.ToTensor(),  # 自动缩放到 0-1
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )] )
    save_path=args.save_path
    data=np.load(npz_path,allow_pickle=True)
    indication=data['indication']
    action=indication.shape[0]//batch_size
    dataset=ImageLoader(npz_path, transform=image_normalize) 
    
    critic=criticNet(config1,config2,sequence,img_sequence).to(device)
    actor=image_transform(config1,config2,sequence).to(device)
    
    done=0 
    for i in range(action):
     algo=environment_model(config1,config2,sequence,img_sequence)
     train_input=DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=2)
     train_input2=DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=2)
     
     for (image,prompts,action),(next_image,next_prompts,next_action) in zip(train_input,train_input2):
      actor_state,critic_state,next_critic_state,present_loss,next_loss=algo.update(image,action,prompts,next_image,next_prompts,next_action,done)  
      reset_(actor,critic)
      actor_save_path=os.path.join(save_path,f"actor_state_dict.pt{i}")
      critic_save_path=os.path.join(save_path,f"critic_state_dict.pt{i}")
      next_critic_save_path=os.path.join(save_path,f"next_critic_state_dict.pt{i}")
      torch.save(actor_state,actor_save_path)
      torch.save(critic_state,critic_save_path)
      torch.save(next_critic_state,next_critic_save_path)
      done+=1/action
def reset_(net1:nn.Module,net2:nn.Module):
    for parameters1,parameters2 in net1.parameters() and net2.parameters():
        parameters1.data.copy_(torch.zeros_like(parameters1.data))
        parameters2.data.copy_(torch.zeros_like(parameters2.data))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz_path",default="J:/image_language_model/image_text.npz")
    p.add_argument("--device",default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--config1",default=load_env_file("J:/image_language_model/config/image_description.yaml"))
    p.add_argument("--config2",default=load_env_file("J:/image_language_model/config/image_text_critic.yaml"))
    p.add_argument("--save_path",type=str, default="J:/image_language_model")
    p.add_argument("--sequence",type=int,default=1)
    p.add_argument("--img_sequence",type=int,default=1)
    p.add_argument("--batch_size",type=int,default=10)
    p.add_argument("--epoches",type=int, default=10)
    args = p.parse_args()
    train(args)
if  __name__=='__main__':
    main()
