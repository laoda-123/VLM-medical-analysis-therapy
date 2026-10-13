import numpy as np 
import matplotlib.pyplot as plt 
from abc import ABC,abstractmethod
import math
class BernoulliBandit:
 def __init__(self,K):
  self.probs=np.random.uniform(size=K)
  self.K=K
  self.best_idx=np.argmax(self.probs)
  self.best_probs=self.probs[self.best_idx]

 def step (self,k):
    if np.random.rand()<self.probs[k]:
      return 1
    else:
      return 0 

#尝试贪婪算法的不同的梯度大小以及种类
class Solver:
  def __init__(self,bandint):
    self.bandint=bandint
    self.counts=np.zeros((self.bandint.K))
    self.regret=0.0
    self.actions=[]
    self.regrets=[]
  def update(self,k):
    self.regret+=self.bandint.best_probs-self.bandint.probs[k] 
    self.regrets.append(self.regret)

  def  run_one_step(self):
   raise NotImplementedError
  
  def  run(self,num_steps):
   for _ in range(num_steps):
    k=self.run_one_step()
    self.actions.append(k)
    self.update(k)

class EpsilonGreedy(Solver):
  def __init__(self,bandint,init_prob=1.0):
    super(EpsilonGreedy,self).__init__(bandint)
    
  #  self.epsilon=epsilon
    self.init_prob=init_prob
    self.estimates=np.array([init_prob]*self.bandint.K)
    self.total_count=0 
  def run_one_step(self):
    self.total_count+=1
    if np.random.rand()<1/self.total_count:
      k=np.random.randint(0,self.bandint.K)
    else:
      k=np.argmax(self.estimates)
    r=self.bandint.step(k)
    self.estimates[k]+=1.0/self.counts[k]*(r-self.estimates[k])
    return k   
  


def plot_solvers(solvers,solver_name):
  for idx,solver in enumerate(solvers):
    timestep=np.arange(len(solver.regrets))
    plt.plot(timestep,solver.regrets,label=solver_name[idx])
    plt.xlabel('time_steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandint' %solvers[0].bandint.K)
    plt.legend()
    plt.show()
np.random.seed(1)
K=10
bandint_10=BernoulliBandit (K=10)
np.random.seed(1)
epsilon_greedy_solver=EpsilonGreedy(bandint_10
)
epsilon_greedy_solver.run(5000)
plot_solvers([epsilon_greedy_solver],["EpsilonGreedy"])