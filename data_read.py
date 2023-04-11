from light_curve import LightCurve
import joblib
from torch.utils.data import Dataset, DataLoader

path_macho = '/Users/francisco/Documents/data/macho/full.pkl'
data_macho = joblib.load(path_macho)

path_asas = '/Users/francisco/Documents/data/asas/full.pkl'
data_asas = joblib.load(path_asas)

path_linear = '/Users/francisco/Documents/data/linear/full.pkl'
data_linear = joblib.load(path_linear)

all_data = data_macho + data_asas + data_linear

class data(Dataset):
    def __init__(self,data):
        super().__init__()