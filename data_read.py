from light_curve import LightCurve
import joblib
from torch.utils.data import Dataset, DataLoader

class LCD(Dataset):
    def __init__(self,data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __get_item__(self,index):
        return self.data[index]
