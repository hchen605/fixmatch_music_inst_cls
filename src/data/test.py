import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
import pandas as pd

def binarize_targets(targets, threshold=0.5):
    targets[targets < threshold] = 0
    targets[targets > 0] = 1
    return targets

class Dataset(Dataset):
    def __init__(self, path, half_precision=False, feat_type='vgg', specaug=False, aug=False):
        #h = h5py.File(path, 'r')
        h = np.load(path, allow_pickle=True)
        self.h = None
        self.h5_path = path
        self.length = h['X'].shape[0]
        #self.H, self.W = h['vggish'][0].shape
        self.labelled = h['Y_mask'][:].sum()
        self.total = np.prod(h['Y_mask'].shape)
        # self.pos_weights = torch.load('./data/pos_weights.pth')
        self.half = half_precision
        self.feat_type = feat_type
        self.specaug = specaug
#         self.wave = np.load('waveshape.npy')
        self.Y_true = h['Y_true'][:]
        self.Y_mask = h['Y_mask'][:]
        h.close()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Lazy opening of h5 file to enable multiple workers
        # in the dataloader init. Only for augmentation though
        if self.h is None:
            print(index)
            #self.h = h5py.File(self.h5_path, 'r', libver='latest', swmr=True)
            self.h = np.load(self.h5_path, allow_pickle=True)
            #self.X = self.h['vggish'][:]/255.0
            self.X = self.h['X'][:]/255.0
            self.Y_true = self.h['Y_true'][:]
            self.Y_mask = self.h['Y_mask'][:]
            self.Y_true[self.Y_mask == 0] = 0.
        # Add routines to decide which data to pick up depending on what augmentation to use
        # In future iterations, also experiment with outputting two augmented versions of X
        t_dtype = torch.float16 if self.half else torch.float32
        if self.feat_type == 'vgg':
            X = self.X[index]
        elif self.feat_type == 'vgg_wave':
            i = np.random.choice(range(7))
            if i == 0:
                X = self.X[index]
            else:
                X = self.wave[index][i-1]
        elif self.feat_type == 'spec':
            X = self.spec[index]
#             if self.specaug:
#                 X = specaugment(X)
        elif self.feat_type == 'audio':
            X = self.audio[index]
        # X = X.reshape(self.H, self.W)
        Y_true = binarize_targets(self.Y_true[index])
        Y_mask = self.Y_mask[index]
        X = torch.tensor(X, requires_grad=False, dtype=t_dtype)
        Y_true = torch.tensor(Y_true.astype(float), requires_grad=False, dtype=t_dtype)
        Y_mask = torch.BoolTensor(Y_mask.astype(bool))
        return X, Y_true, Y_mask

    def terminate(self):
        if self.h is not None:
            self.h.close()
            

abs_path = '/home/hc605/7100_21_fall/audio_missing_labels/data'
OPENMIC = np.load(os.path.join(abs_path, 'openmic-2018.npz'), allow_pickle=True)
print(list(OPENMIC.keys()))
X, Y_true, Y_mask, sample_key = OPENMIC['X'], OPENMIC['Y_true'], OPENMIC['Y_mask'], OPENMIC['sample_key']
print('---- OPENMIC.npz ----')
print(X.shape)

OPENMIC = np.load(os.path.join(abs_path, 'train.npz'), allow_pickle=True)
print(list(OPENMIC.keys()))
X, Y_true, Y_mask, sample_key = OPENMIC['X'], OPENMIC['Y_true'], OPENMIC['Y_mask'], OPENMIC['sample_key']
print('---- train.npz ----')
print(X.shape)

OPENMIC = np.load(os.path.join(abs_path, 'test.npz'), allow_pickle=True)
print(list(OPENMIC.keys()))
X, Y_true, Y_mask, sample_key = OPENMIC['X'], OPENMIC['Y_true'], OPENMIC['Y_mask'], OPENMIC['sample_key']
print('---- test.npz ----')
print(X.shape)

print('train_val.split')
train_val_split = np.load(os.path.join(abs_path, 'train_val.split'))
print(train_val_split['train'])
print(train_val_split['train'].shape)
print(train_val_split['val'])  
print(train_val_split['val'].shape)  

full_dataset = Dataset(os.path.join(abs_path, 'openmic-2018.npz'))

train_dataset_all = Dataset(os.path.join(abs_path, 'train.npz'))
test_dataset = Dataset(os.path.join(abs_path, 'test.npz'))



split_train = pd.read_csv(os.path.join(abs_path, 'split01_train.csv'), 
                      header=None, squeeze=True)
split_test = pd.read_csv(os.path.join(abs_path, 'split01_test.csv'), 
                     header=None, squeeze=True)

train_set = set(split_train)
test_set = set(split_test)

#print(train_set)
#print(test_set)

abs_path = '/home/hc605/7100_21_fall/audio_missing_labels/data'
OPENMIC = np.load(os.path.join(abs_path, 'openmic-2018.npz'), allow_pickle=True)
#print(list(OPENMIC.keys()))
X, Y_true, Y_mask, sample_key = OPENMIC['X'], OPENMIC['Y_true'], OPENMIC['Y_mask'], OPENMIC['sample_key']

idx_train, idx_test = [], []

for idx, n in enumerate(sample_key):
    if n in train_set:
        idx_train.append(idx)
    elif n in test_set:
        idx_test.append(idx)
    else:
        # This should never happen, but better safe than sorry.
        raise RuntimeError('Unknown sample key={}! Abort!'.format(sample_key[n]))


# Finally, cast the idx_* arrays to numpy structures
idx_train = np.asarray(idx_train)
idx_test = np.asarray(idx_test)
print('---- idx ----')
print(idx_train)
print(idx_test)
print(idx_train.size)
print(idx_test.size)

print(X[29])
print(Y_true[29])
print(Y_mask[29])

train_val_split = np.load(os.path.join(abs_path, 'train_val.split'))
#print(train_val_split)

#train_dataset = Subset(full_dataset, idx_train[0:9109])
train_dataset = Subset(train_dataset_all, train_val_split['train'])
#val_dataset = Subset(full_dataset, idx_train[9109:])
val_dataset = Subset(train_dataset_all, train_val_split['val'])
#test_dataset = Subset(full_dataset, idx_test)
#test_dataset = HDF5Dataset(os.path.join(abs_path, 'openmic_test.h5'))

print('---- train_dataset[0] ----')
X, Y_true, Y_mask = train_dataset[1]
print(X)
print(Y_true)
print(Y_mask)

print('---- test_dataset[0] ----')
X, Y_true, Y_mask = test_dataset[1]
print(X)
print(Y_true)
print(Y_mask)

#batch_size = config['hparams']['batch_size']
#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
#val_loader = torch.utils.data.DataLoader(val_dataset, batch_size)
#test_loader = torch.utils.data.DataLoader(test_dataset, batch_size)

'''
abs_path = config['data_path']
full_dataset = HDF5Dataset(os.path.join(abs_path, 'openmic_train.h5'))
train_val_split = np.load(os.path.join(abs_path, 'train_val.split'))

train_dataset = Subset(full_dataset, train_val_split['train'])
val_dataset = Subset(full_dataset, train_val_split['val'])
test_dataset = HDF5Dataset(os.path.join(abs_path, 'openmic_test.h5'))

batch_size = config['hparams']['batch_size']
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size)
'''




            
class HDF5Dataset(Dataset):
    def __init__(self, h5_path, half_precision=False, feat_type='vgg', specaug=False, aug=False):
        h = h5py.File(h5_path, 'r')
        self.h = None
        self.h5_path = h5_path
        self.length = h['vggish'].shape[0]
        self.H, self.W = h['vggish'][0].shape
        self.labelled = h['Y_mask'][:].sum()
        self.total = np.prod(h['Y_mask'].shape)
        # self.pos_weights = torch.load('./data/pos_weights.pth')
        self.half = half_precision
        self.feat_type = feat_type
        self.specaug = specaug
#         self.wave = np.load('waveshape.npy')
        self.Y_true = h['Y_true'][:]
        self.Y_mask = h['Y_mask'][:]
        h.close()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Lazy opening of h5 file to enable multiple workers
        # in the dataloader init. Only for augmentation though
        if self.h is None:
            self.h = h5py.File(self.h5_path, 'r', libver='latest', swmr=True)
            self.X = self.h['vggish'][:]/255.0
            self.Y_true = self.h['Y_true'][:]
            self.Y_mask = self.h['Y_mask'][:]
            self.Y_true[self.Y_mask == 0] = 0.
        # Add routines to decide which data to pick up depending on what augmentation to use
        # In future iterations, also experiment with outputting two augmented versions of X
        t_dtype = torch.float16 if self.half else torch.float32
        if self.feat_type == 'vgg':
            X = self.X[index]
        elif self.feat_type == 'vgg_wave':
            i = np.random.choice(range(7))
            if i == 0:
                X = self.X[index]
            else:
                X = self.wave[index][i-1]
        elif self.feat_type == 'spec':
            X = self.spec[index]
#             if self.specaug:
#                 X = specaugment(X)
        elif self.feat_type == 'audio':
            X = self.audio[index]
        # X = X.reshape(self.H, self.W)
        Y_true = binarize_targets(self.Y_true[index])
        Y_mask = self.Y_mask[index]
        X = torch.tensor(X, requires_grad=False, dtype=t_dtype)
        Y_true = torch.tensor(Y_true.astype(float), requires_grad=False, dtype=t_dtype)
        Y_mask = torch.BoolTensor(Y_mask.astype(bool))
        return X, Y_true, Y_mask

    def terminate(self):
        if self.h is not None:
            self.h.close()
            
def audio_to_vggish(audio):
    feats = VGGish(audio)
    return feats/255.0