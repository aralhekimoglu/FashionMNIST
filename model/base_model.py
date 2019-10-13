from __future__ import print_function, absolute_import
import os

import torch
from torch import nn
from torch.optim import lr_scheduler

from model.networks.basenet import BaseNet
from utils.utils import print_parameter_count

class BaseModel(object):
    def __init__(self,args):
        self.args = args
        self._init_networks()
        self._init_optimizers()
        
    def _init_networks(self):
        self.net = BaseNet(self.args)
        print_parameter_count(self.net)
        self.net = torch.nn.DataParallel(self.net).cuda()

    def _init_optimizers(self):
        self.loss_fn = nn.CrossEntropyLoss()
        if self.args.optimizer=='adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(),lr=self.args.lr)
        else:
            self.optimizer = torch.optim.SGD(self.net.parameters(),lr=self.args.lr, momentum=0.9,weight_decay=5e-4)

        if self.args.lr_scheduler == 'step':
            self.scheduler = lr_scheduler.StepLR(self.optimizer,step_size=10,gamma=0.1)
        else:
            self.scheduler = lr_scheduler.ExponentialLR(self.optimizer,gamma=0.99)

    def train(self):
        self.net.train()
    
    def eval(self):
        self.net.eval()
        
    def set_input(self,data):
        img,labels = data
        self.img = img.cuda()
        self.labels = labels.cuda()
        
    def forward(self):
        self.logits = self.net(self.img)
    
    def optimize_parameters(self):
        self.train()
        self.forward()
        self.optimizer.zero_grad()
        
        self.loss = self.loss_fn(self.logits, self.labels)
        
        self.loss.backward()
        self.optimizer.step()
        
    def get_loss(self):
        return self.loss.data

    def save_model(self,epoch):
        save_name = '%s_net.pth' % (epoch)
        save_path = os.path.join(self.args.ckpt_dir,save_name)
        torch.save(self.net.module.state_dict(),save_path)

    def update_lr(self):
        self.scheduler.step()