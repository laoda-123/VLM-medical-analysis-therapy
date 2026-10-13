import torch  
import torch.nn as nn 
import torch.nn.functional as F 
from net.image_recognition import image_transform 
from net.image_text_critic import criticNet

from torch.distributions import Normal
import numpy as np 
class environment_model():
    def __init__(self,config1,config2,sequence,img_sequence):
        super(environment_model, self).__init__()
        self.actor =image_transform(config1,config2,sequence)
        self.critic1=criticNet(config1,config2,sequence,img_sequence)
        self.critic2=criticNet(config1,config2,sequence,img_sequence)
        self.target_next_critic=criticNet(config1,config2,sequence,img_sequence)
        self.target_critic=criticNet(config1,config2,sequence,img_sequence)
        self.gamma=config2['parameters']['gamma']
        
        self.image_text_learning_rate=config2['parameters']['lr']
        self.actor_lr=config2['actor_lr']
        
        self.critic_lr=config2['critic_lr']
        self.target_critic_lr=config2['target_critic_lr']
        self.alpha=torch.tensor(np.log(0.001),requires_grad=True)
        self.alpha_optimizer=torch.optim.Adam(self.alpha,lr=self.image_text_learning_rate)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=self.actor_lr)
        
        self.critic_optimizer=torch.optim.Adam(self.critic1.parameters(),lr=self.critic_lr)
        self.next_critic_optimizer=torch.optim.Adam(self.critic2.parameters(),lr=self.critic_lr)
        self.target_critic_optimizer=torch.optim.Adam(self.target_critic.parameters(),lr=self.target_critic_lr)
        self.next_critic_optimizer=torch.optim.Adam(self.target_next_critic.parameters(),lr=self.target_critic_lr)
        self.batch_size=config2['parameters']['batch-size']
        self.epoches=config2['train_epoches']
    def calculate_target(self,state,prompts,action,next_state,next_action,next_prompts,done):
        actor_result =self.actor(state,prompts)
        actor_mean =torch.mean(actor_result,dim=0)
        actor_std=torch.std(actor_result,dim=0)
        action_mean=torch.mean(action,dim=0)
        action_std=torch.std(action,dim=0 )
        mean_delta=action_mean-actor_mean 
        std_delta=torch.abs(action_std-actor_std)
        sample=Normal(mean_delta ,torch.abs(std_delta)).sample((self.batch_size,))
        output=torch.clip_(sample,mean_delta-std_delta,mean_delta+std_delta)
        present_target=actor_result*(1-done)+output*self.gamma*self.alpha*done
        
        real_mean=torch.mean(next_action,dim=0)
        real_std=torch.std(next_action,dim=0)
        next_pred_action=self.critic1(next_state,action)
        actor_action=self.actor(next_state,next_prompts)
        next_mean =torch.mean(actor_action,dim=0)
        next_std =torch.std(actor_action,dim =0)
        
        pred_mean=torch.mean(next_pred_action,dim =0)
        pred_std=torch.std(next_pred_action,dim=0)
        actor_loss= F.mse_loss(next_mean,real_mean)
        pred_actor_loss=F.mse_loss(pred_mean,real_mean)
        if pred_actor_loss>actor_loss:
            self.next_pred_action =actor_action
            mean_delta= real_mean-next_mean 
            std_delta=real_std-next_std 
            sample=torch.clip_(Normal(mean_delta,std_delta).sample((self.batch_size,)),mean_delta-torch.abs(std_delta),mean_delta+torch.abs(std_delta))
            target=actor_action*(1-done)+self.gamma*self.alpha*sample*done
        else:
            self.next_pred_action =next_pred_action
            mean_delta=real_mean-pred_mean
            
            std_delta =real_std-pred_std
            sample=torch.clip_(Normal(mean_delta,std_delta).sample((self.batch_size,)),mean_delta-torch.abs(std_delta),mean_delta+torch.abs(std_delta))
            target=next_pred_action*done+self.gamma*self.alpha*sample*(1-done)
        return present_target,target,self.next_pred_action,actor_result
    def update(self,state,action,prompts,next_state,next_prompts,next_action,done): # only  for image integration for 2 step/scenes 
        
        present_target,next_target,next_pred_action,actor_result=self.calculate_target(state,prompts,action,next_state,next_action,next_prompts,done)
        present_loss,next_loss=F.mse_loss(actor_result,present_target),F.mse_loss(next_pred_action,next_target)
        for i in range(self.epoches):
         present_loss.backward()
         next_loss.backward()
         self.actor_optimizer.zero_grad()
         self.actor_optimizer.step()
         self.alpha_optimizer.zero_grad()
         self.alpha_optimizer.step()
         self.critic_optimizer.zero_grad()
         self.critic_optimizer.step() 
         present_target,next_target,next_pred_action,actor_result=self.calculate_target(state,prompts,action,next_state,next_action,next_prompts,done)
        
        return self.actor.state_dict(),self.critic1.state_dict(),self.critic2.state_dict(),present_loss,next_loss


def reset_parameters(net1,net2):
    for data1,data2 in net1.parameters(), net2.parameters():
        data1.data=torch.zeros_like(data1)
        data2.data=torch.zeros_like(data2)

