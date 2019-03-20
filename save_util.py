import torch
import torch.nn as nn
import numpy as np
def save_state(net, opt, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, seed, fname):
    # save model state, optimizer state, train_loss, val_loss, random_seed
    states = {
        'state_dict': net.state_dict(),
        'optimizer': opt.state_dict(),
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'seed': seed
    }
    torch.save(states, fname)

def load_net_state(net, fname):
    checkpoint = torch.load(fname)
    net.load_state_dict(checkpoint['state_dict'])

def load_opt_state(opt, fname):
    checkpoint = torch.load(fname)
    opt.load_state_dict(checkpoint['optimizer'])

def load_eval(fname):
    checkpoint = torch.load(fname)
    return checkpoint['train_loss'], checkpoint['train_acc'], checkpoint['val_loss'], \
           checkpoint['val_acc'], checkpoint['test_loss'], checkpoint['test_acc']

def load_seed(fname):
    checkpoint = torch.load(fname)
    return checkpoint['seed']
