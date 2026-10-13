import collections
from tqdm import tqdm
import torch 
import torch.nn as nn 
from envs.env_builder import load_env_file
import random


class replay_buffer():
   def __init__(self,state,next_state,action,reward,done,config):
     self.state=state
     self.next_state=next_state
     self.action=action
     self.reward=reward
     self.done=done
     maxlen=config['max_episode_scene_frame']
     self.buffer=collections.deque(maxlen=maxlen) # select the frame axis, and the shot concvertion scene
   def add(self,state,next_state,action,reward,done):
      self.buffer.append((state,next_state,action,reward,done))
   def sample(self,batch_size):
      transition=random.sample(self.buffer,batch_size)
      state,action,next_state,reward,done=zip(*transition)
      return state,action,next_state,reward,done
   def delete(self,pool_len,batch_size):
      for i in range(batch_size):
         self.buffer.remove(self.buffer[i])
   def size(self):
      return len(self.buffer)
   