import os

import torch
from src.utils import encode_aa, get_backbone
from torch.utils.data import Dataset
from tqdm import tqdm


class PDBDataset(Dataset):
    def __init__(self, data_path: str, transforms=None):
        """
        Args:
            data (array-like): The input data.
            labels (array-like): The labels corresponding to the data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_path = data_path
        self.data = self._prepare_x_and_h(data_path)
        self.transforms = transforms
        
    def _prepare_x_and_h(self, data_path):
        """
        Prepares the input features and labels for protein data.
        Args:
            data_path (str): The path to the directory containing protein data files.
        Returns:
            list: A list of dictionaries, each containing:
                - "positions" (torch.Tensor): Tensor of backbone atom coordinates.
                - "i_seq" (np.ndarray): Encoded amino acid sequence.
        """
        
        output = []
        for file in tqdm(os.listdir(data_path)):
            full_path = os.path.join(data_path, file)
            try:
                protein_dict = {}
                frames, seq = get_backbone(full_path)
                x_ca = frames[0][:,1,:]
                i_seq = encode_aa(seq[0])
                protein_dict["positions"] = torch.Tensor(x_ca)
                protein_dict["i_seq"] = i_seq
                output.append(protein_dict)
            except:
                continue
        return output

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transforms:
            return self.transforms(self.data[idx])
        return self.data[idx]