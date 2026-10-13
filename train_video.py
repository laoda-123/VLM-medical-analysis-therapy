import  matplotlib.pyplot as plt
from net.video_recognition import stable_diffusion
from net.UNET import UNetDenoise 
from net.value_critic_Net import StepNet
import torch

from PIL import Image
import argparse
import numpy as np 
import torch.nn as nn  
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection  import train_test_split
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
from envs.env_builder import load_env_file
from net.target_critic import Trajectory_train
import random 
from train_data import reset_
seed=np.random.seed(42)
class text_image(Dataset): # use batch to train/ test is another use
   def __init__(self,npz_path:str,train_factor:float=0.6,seed:int=0):
      self.data=np.load(npz_path)
      self.items=[] 
      train_input,train_target=self.data["description"],self.data["image"]
      self.transform = transforms.Compose([
      transforms.ToTensor(),  # 自动缩放到 0-1
      transforms.Normalize(
        mean=[0.485, 0.456, 0.406,0.5],
        std=[0.229, 0.224, 0.225,0.0004]
      )] ) # half apparent/opaque

      for i in range(len(train_input)):
           self.items.append((train_input[i],train_target[i]))
      
      
   def  __len__(self):
      return len(self.items)
   def __getitem__(self,i):
      text,image=self.items[i]
      img=Image.open(image).convert("RGBA")

      return  torch.as_tensor(text,dtype=torch.float32),self.transform(img)
def train(args):
    done=0
    device = args.device 
    npz_path=args.npz_path 
    image=np.load(npz_path) # image is  list type
    train_epoches=args.train_epoches
    config1=load_env_file(args.config1_path)
    config2=load_env_file(args.config2_path)
    sequence=args.sequence 
    json_path=args.json_path
    save_folder=args.save_folder 
    batch_size=args.batch_size
    agent=Trajectory_train(config1,config2,sequence,device)
    prev_loss=[]
    pre=[]
    ds=text_image(npz_path)
   
    action=len(image)//batch_size
    
    for i in range (action):
      dataloader=DataLoader(ds,batch_size=batch_size ,shuffle=False)
      dataloader1=DataLoader(ds,batch_size=batch_size,shuffle=False)
      for (previous_state,previous_action),(state,action) in zip(dataloader,dataloader1) :
       actor_state_dict,critic_state_dict,target_critic_state,previous_actor_loss,present_loss= agent.update(previous_state,previous_action,state,action,done)
       actor_kept_path=save_folder+f"/actor/actor{i}.pt"
       critic_kept_path=save_folder+f"/critic/critic{i}.pt"
       target_critic_kept_path=save_folder+f"/target_critic/tc{i}.pt"
      torch.save(actor_state_dict,actor_kept_path)
      torch.save(critic_state_dict,critic_kept_path)
      torch.save(target_critic_state,target_critic_kept_path)
      prev_loss.append(previous_actor_loss )
      pre.append(present_loss)
      stepNet=StepNet(config1,config2,sequence,device)
      actor=stable_diffusion(config1,config2,sequence,device)
      reset_(stepNet,actor)
    if prev_loss:
      plt.xlim(0,action)
      plt.ylim(-1,1)
      plt.xlabel("recursion_epochs")
      plt.ylabel("prev_actor_loss")
      plt.plot(np.arange(action)+1,prev_loss)
      plt.show() 
    if pre:
      plt.xlim(0,action )
      plt.ylim(-1,1)
      plt.xlabel("recursion_epochs")
      plt.ylabel("present_loss")
      plt.plot(np.arange(action),present_loss)
      plt.show()
    plt.close()
def main():
  p = argparse.ArgumentParser()
  p.add_argument("--device",default=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
  p.add_argument("--npz_path",type=str,default="")
  p.add_argument("--config1_path",type=str,default="J:/image_language_model/config/image_description.yaml"
                 )
  p.add_argument("--config2_path",type=str,default="J:/image_language_model/config/text_image.yaml")
  p.add_argument("--sequence" ,type=int,default=1)
  p.add_argument("--save_folder",type=str,default="I:/image_language_model") 
  p.add_argument("--json_path",type=str,default="I:/image_language_model/model.json")
  p.add_argument("--train_epoches",type=int,default=10)
  p.add_argument("--batch_size",type=int,default=20)
  args=p.parse_args()
  train(args)
if __name__=='__main__':
   main()
   