import torch
from torch.utils.data import Dataset

class MedicalDataset(Dataset):
    def __init__(self, encodings, lab_results, labels):
        self.encodings = encodings
        self.lab_results = lab_results
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'lab_results': torch.tensor(self.lab_results[idx]),
            'labels': torch.tensor(self.labels[idx])
        }
        return item

    def __len__(self):
        return len(self.labels)
