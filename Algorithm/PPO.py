import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from  net.value_critic_Net import StepNet
from  envs.env_builder import load_env_file
from util.compute_advantange import compute_advantage
import torch
class PPO():
    def __init__(self,config1,config2,word,sequence):
        self.critic=StepNet(config1,config2,word,sequence) # this config refers to absolute/relative path
        self.actor=StepNet(config1,config2,word,sequence)
        self.config1=load_env_file(config1)
        self.config2=load_env_file(config2)
        self.lr=self.config1['parameters']['lr']
        self.actor_optimizer=torch.optim.AdamW(self.actor.parameters(),self.lr)
        self.critic_optimizer=torch.optim.AdamW(self.critic.parameters(),self.lr )
        self.gamma=self.config2['gamma']
        self.lamda=self.config2['lambda']
        self.epochs=self.config2['epochs']
        self.eps=self.config2['eps']
    def take_action(self,state):
     state=torch.tensor(state,dtype=torch.float)
     action=self.actor(state)
     action_dist=torch.distributions.Categorical(action)
     action=action_dist.sample()
     return action
    def update(self,state,next_state,action,reward,done):
     state=torch.tensor(state,dtype=torch.float)
     next_state=torch.tensor(next_state,dtype=torch.float)
     action=torch.tensor(action ,dtype=torch.float)
     done=torch.tensor(done,dtype=torch.float)
     rewards=torch.tensor(reward,dtype=torch.float)
     td_targets=rewards+self.gamma*self.critic(next_state)*(1-done) 
     td_delta=td_targets-self.critic(state)
     advantage=compute_advantage(self.gamma,self.lamda,td_delta)
     old_log_probs=torch.log(self.actor(state).gather(1,action)).detach()
     for _ in range(self.epochs):
      log_probs=torch.log(self.actor(state).gather(1,action))
      ratio=torch.exp(log_probs-old_log_probs)
      surr1=ratio*advantage
      surr2=torch.clamp(ratio,1-self.eps,1+self.eps)
      actor_loss=torch.mean(-torch.min(surr1,surr2))
      critic_loss=torch.mean(F.mse_loss(self.critic(state),td_targets.detach()))
      self.actor_optimizer.zero_grad()
      self.critic_optimizer.zero_grad()
      actor_loss.backward()
      critic_loss.backward() 
      self.actor_optimizer.step()
      self.critic_optimizer.step()
     return reward,done

