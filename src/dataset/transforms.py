import torch
from tqdm import tqdm


class NormalizeCoordinates:
    def __init__(self):
        pass
    def __call__(self, data):
        pos = data["positions"]
        pos = pos - pos.mean(dim=0)
        data["positions"] = pos
        return data

class PadDatasetTransform:
    def __init__(self, max_i_seq_len=500):
        self.max_i_seq_len = max_i_seq_len

    def __call__(self, data):
        i_seq = data["i_seq"]
        i_seq_len = len(i_seq)
        if self.max_i_seq_len > i_seq_len:
            i_seq = torch.cat([i_seq, torch.Tensor([0] * (self.max_i_seq_len - i_seq_len))])
            data["i_seq"] = i_seq
        pos = data["positions"]
        pos_len = len(pos)
        if self.max_i_seq_len > pos_len:
            pos_pad = torch.Tensor([[0, 0, 0]] * (self.max_i_seq_len - pos_len))
            pos = torch.vstack([pos, pos_pad])
            data["positions"] = pos
        return data


# Example usage:
# from torchvision.transforms import Compose
# transform = Compose([PadDatasetTransform(max_i_seq_len=500)])
# transformed_dataset = [transform(data) for data in dataset]

# Example usage:
# transform = PadDatasetTransform(max_i_seq_len=500)
# transformed_dataset = transform(dataset)