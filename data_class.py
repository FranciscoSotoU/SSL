from light_curve import LightCurve

from torch.utils.data import Dataset
import torch

class data_LC(Dataset):
    
    def __init__(self,input,label):
        self.data = torch.tensor(input)
        self.labels = torch.tensor(label)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index],self.labels[index]
    