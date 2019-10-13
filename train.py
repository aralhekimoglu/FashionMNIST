from __future__ import print_function, absolute_import
import os
import sys
import time

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.logging import Logger
from utils.arguments import Arguments
from utils.evaluator import Evaluator
from model.base_model import BaseModel
from utils.data.dataloader import get_data

def main(args):
    sys.stdout = Logger(args.log_dir)
    
    train_loader,test_loader = get_data(args)
    model = BaseModel(args)
    evaluator = Evaluator(model = model,data_loader = test_loader)
    
    best_acc = evaluator.evaluate()
    
    accuracies = [best_acc]
    losses = []
        
    for e in range(1,args.epochs+1):
        epoch_loss = 0
        print("Epoch",e)
        for data in tqdm(train_loader):
            model.set_input(data)
            model.optimize_parameters()
            epoch_loss += model.get_loss()
            
        print("Epoch finished with loss",epoch_loss)
        losses.append(epoch_loss)
        
        if e % args.eval_step == 0 :
            acc = evaluator.evaluate()
            accuracies.append(acc)
            best_acc = max(acc,best_acc)
            print("[Epoch {}] Accuracy:{:.2f}, Best Accuracy:{:.2f}".format(e,acc,best_acc))
        
        if e % args.save_step == 0:
            model.save_model(e)

        model.update_lr()

        plt.figure()
        plt.plot(range(len(losses)), losses)
        plt.xlabel('Epochs')
        plt.ylabel('Training Loss')
        plt.savefig(os.path.join(args.exp_dir,'losses.png'))
        
        plt.figure()
        plt.plot(range(len(accuracies)), accuracies)
        plt.xlabel('Epochs')
        plt.ylabel('Test Accuracy')
        plt.savefig(os.path.join(args.exp_dir,'accuracies.png'))
        
if __name__ == '__main__':
    args = Arguments().parse()
    main(args)