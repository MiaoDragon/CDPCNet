"""
ref: UCSD CSE253 W19 course hw3
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler

class ComplexDataset(Dataset):
    def __init__(self, path, name, batch, minibatch):
        """
        # Input:
        - path: path to load data
        - file in format: cP1_[idx], cP2_[idx], P1_[idx], P2_[idx]
        ! cP or P depends on name
        """
        self.path = path
        self.batch = batch
        self.name = name
        self.minibatch = minibatch
    def __len__(self):
        return self.batch
    def __getitem__(self, ind):
        batch_idx = ind // self.minibatch
        mini_idx = ind % self.minibatch  # index inside each minibatch
        P1 = np.load(self.path+self.name+'1_%d.npy' % (batch_idx))[mini_idx]
        P2 = np.load(self.path+self.name+'2_%d.npy' % (batch_idx))[mini_idx]
        P1 = torch.from_numpy(P1).type(torch.FloatTensor)
        P2 = torch.from_numpy(P2).type(torch.FloatTensor)
        if self.name == 'cP':
            y = 1  # collide
        else:
            y = 0
        return P1, P2, y

def create_split_loaders(path, total_size, mini_size, batch_size, seed,
                         p_val=0.1, p_test=0.2, extras={},
                         train_sampler=SubsetRandomSampler,
                         val_sampler=SubsetRandomSampler,
                         test_sampler=SubsetRandomSampler,):
    dataset_list = []
    for name in ['cP', 'P']:
        dataset = ComplexDataset(path=path, name=name, batch=total_size//2, minibatch=mini_size)
        dataset_list.append(dataset)
    all_indices = list(range(total_size))
    np.random.seed(seed)
    np.random.shuffle(all_indices)
    # Create the validation split from the full dataset
    val_split = int(np.floor(p_val * total_size))
    train_ind, val_ind = all_indices[val_split :], all_indices[: val_split]
    # Separate a test split from the training dataset
    test_split = int(np.floor(p_test * len(train_ind)))
    train_ind, test_ind = train_ind[test_split :], train_ind[: test_split]
    #print(train_ind)
    #print(val_ind)
    #print(test_ind)
    dataset = ConcatDataset(dataset_list)
    # Use the SubsetRandomSampler as the iterator for each subset
    sample_train = train_sampler(train_ind)
    sample_test = test_sampler(test_ind)
    sample_val = val_sampler(val_ind)

    num_workers = 0
    pin_memory = True
    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]

    # Define the training, test, & validation DataLoaders
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=sample_train, num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(dataset, batch_size=batch_size,
                             sampler=sample_test, num_workers=num_workers,
                              pin_memory=pin_memory)

    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=sample_val, num_workers=num_workers,
                              pin_memory=pin_memory)

    # Return the training, validation, test DataLoader objects
    return (train_loader, val_loader, test_loader)
