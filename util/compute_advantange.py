import torch 

def compute_advantage(gamma,lamda,td_delta):
   td_delta=td_delta.detach().numpy()
   advantage_list =[] 
   advantage=0
   for delta  in td_delta[::-1]:
     advantage=gamma*lamda*advantage+delta
     advantage_list.append(advantage)
   advantage_list.reverse()
   return torch.tensor(advantage_list,dtype=torch.float)