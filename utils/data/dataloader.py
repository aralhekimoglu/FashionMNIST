import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import FashionMNIST

import utils.data.transforms as DT

augmentation_funcs = {'random_hflip':DT.RandomHorizontalFlip(),
                      'random_vflip':DT.RandomVerticalFlip(), 
                      'random_rotate':DT.RandomRotation(),
                      'random_erase':DT.RandomErasing(),
                      'random_trans':DT.RandomTranslation()}
                     
def get_data(args):
    normalizer = T.Normalize(mean=[0.1307],std=[0.3081])

    train_transforms = []

    for augm_str in args.augmentations:
        if augm_str=='random_erase':continue
        train_transforms.append(augmentation_funcs[augm_str])
    
    train_transforms.append(T.ToTensor())
    train_transforms.append(normalizer)
    if 'random_erase' in args.augmentations:train_transforms.append(augmentation_funcs['random_erase'])

    train_transforms = T.Compose(train_transforms)
    test_transforms = T.Compose([T.ToTensor(),normalizer])

    train_dataset = FashionMNIST(args.data_dir, train=True, transform=train_transforms, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.train_batch_size,shuffle=True,num_workers=args.workers)

    test_dataset = FashionMNIST(args.data_dir, train=False, transform=test_transforms, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.test_batch_size,shuffle=False,num_workers=args.workers)
    
    return train_loader,test_loader