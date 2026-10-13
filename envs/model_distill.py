import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import DistilBertTokenizer, DistilBertModel, CLIPModel, CLIPProcessor
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

#select arbitary model  from the model config
