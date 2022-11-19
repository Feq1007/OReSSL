import torch

class DataSet(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.true_label = labels[:,0]
        self.semi_label = labels[:,1]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.true_label[index], self.semi_label[index]