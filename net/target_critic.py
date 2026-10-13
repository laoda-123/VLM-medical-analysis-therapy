import torch 
import torch.nn as nn 
import torch.nn.functional as F
from net.video_recognition import stable_diffusion
from net.value_critic_Net import StepNet
import numpy as np
from torch.distributions import Normal 
class Trajectory_train():
    def __init__(self,config1,config2,sequence,device): # only for static image batches not for video pipeline
     super(Trajectory_train,self).__init__()
     self.device=device
     self.actor=stable_diffusion(config1,config2,sequence,device)
     self.critic=StepNet(config1,config2,sequence,device)
     self.target_critic1=StepNet(config1,config2,sequence,device)
   
     self.gamma=config2['gamma']
     self.actor_lr=config2['actor_lr']
     self.critic_lr=config2['critic_lr']
     self.alpha_lr=config2['alpha_lr']
     self.target_critic_lr=config2['target_critic_lr']    
     self.sigma =config2['sigma']
     self.alpha =torch.tensor(np.log(config2['alpha']),requires_grad=True)
     self.alpha_optimizer=torch.optim.Adam(self.alpha,self.alpha_lr)
     self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),self.actor_lr)
     self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),self.critic_lr)
     self.target_critic_optimizer1=torch.optim.Adam(self.target_critic1.parameters(),self.target_critic_lr)

     self.batch_size=config2['parameters']['batch_size']
     self.epoches=config2['train_epoches']
    def  soft_update(self,net,net2,done):
     for parameters1,parameters in net.data and net2.data:
      net2.data=parameters1*done+parameters*(1-done)    
      
     return net2
    def update(self,previous_state,previous_action,state,action,done):
        previous_state=torch.as_tensor(previous_state,dtype=torch.float).to(self.device)
        previous_action=torch.as_tensor(previous_action,dtype=torch.float).to(self.device)
        state,action =torch.as_tensor(state).to(self.device),torch.as_tensor(action).to(self.device)
        prev_pred_action=self.actor(previous_state)
        prev_pred_mean=torch.mean(prev_pred_action, dim=0)
        prev_pred_std=torch.std(prev_pred_action,dim=0)
        prev_mean=torch.mean(previous_action,dim=0)
        prev_std=torch.std(previous_action,dim=0)
        prev_delta_mean=prev_mean-prev_pred_mean
        prev_delta_std =prev_std-prev_pred_std
        center_mean=prev_delta_mean.unsqueeze(0).expand(self.batch_size,-1,-1,-1)
        center_std=prev_delta_std.unsqueeze(0).expand(self.batch_size,-1,-1,-1)
        sample=Normal(prev_delta_mean,torch.abs(prev_delta_std) ).sample((self.batch_size,))
        clip_sample=torch.clip_(sample,center_mean-center_std,center_mean+center_std)
        
        for i in range(self.epoches):
          target_train_target=clip_sample*(1-done)+self.alpha*prev_pred_action*done
          prev_actor_loss=F.mse_loss(prev_pred_action,target_train_target)
          self.actor_optimizer.zero_grad()
          prev_actor_loss.backward(retain_graph=True)
          self.actor_optimizer.step()
          self.alpha_optimizer.zero_grad()
          self.alpha_optimizer.step()
          done+=1/self.epoches
            
        
            
        present_pred_action=self.critic(state,prev_pred_action)
        present_actor_action=self.actor(state)
        present_mean,present_std=torch.mean(action,dim=0),torch.std(action,dim=0)
        present_pred_mean=torch.mean(present_pred_action,dim=0)
        present_pred_std=torch.std(present_pred_action,dim=0)
        present_actor_mean=torch.mean(present_actor_action,dim=0)
        present_actor_std=torch.std(present_actor_action,dim=0)
        
        present_critic_delta_mean=present_mean-present_pred_mean
        present_critic_delta_std=present_std-present_pred_std
        present_critic_mean_center=present_critic_delta_mean.unsqueeze(0).expand(self.batch_size,-1,-1,-1)
        present_critic_std_center=present_critic_delta_std.unsqueeze(0).expand(self.batch_size,-1,-1, -1)
        
        
        present_actor_delta_mean=present_mean-present_actor_mean
        present_actor_delta_std=present_mean-present_actor_std
        present_actor_mean_center=present_actor_delta_mean.unsqueeze(0).expand(self.batch_size,-1,-1,-1)
        present_actor_std_center=present_actor_delta_std.unsqueeze(0).expand(self.batch_size,-1,-1,-1)
        loss=F.mse_loss(present_pred_action,action)
        actor_loss=F.mse_loss(present_actor_action,action)    
        
        
        for i in range(self.epoches):
         if torch.abs(loss)>torch.abs(actor_loss):
          present_pred_action=present_actor_action
          sample=Normal(present_actor_delta_mean,torch.abs(present_actor_delta_std)).sample((self.batch_size,))
          clip_sample=torch.clip_(sample,present_actor_mean_center-present_actor_std_center,present_actor_mean_center+present_actor_std_center)
         else:
          sample=Normal(present_critic_delta_mean,present_critic_delta_std).sample((self.batch_size,))
          clip_sample=torch.clip_(sample,present_critic_mean_center-torch.abs(present_critic_std_center),present_critic_mean_center+torch.abs(present_critic_std_center))
         train_target=clip_sample*(1-done)+self.gamma*self.alpha*present_pred_action*done
         present_loss=F.mse_loss(present_pred_action,train_target)
         present_loss.backward()
         self.actor_optimizer.zero_grad()
         self.critic_optimizer.zero_grad()
         self.actor_optimizer.step()
         self.critic_optimizer.step()
         self.alpha_optimizer.zero_grad()
         self.alpha_optimizer.step()
         done+=1/self.epoches
        
        target_critic_=action-prev_actor_loss*torch.randn_like(action)
        overall_ids=self.target_critic1(previous_state,previous_action)
        loss=F.mse_loss(overall_ids,target_critic_)
        loss.backward()
        self.target_critic_optimizer1.zero_grad()
        self.target_critic_optimizer1.step()
        
        return self.actor.state_dict(),self.critic.state_dict(), self.target_critic1.state_dict(),prev_actor_loss,present_loss