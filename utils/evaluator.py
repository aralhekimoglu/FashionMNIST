from __future__ import print_function, absolute_import

import torch
from tqdm import tqdm

class Evaluator(object):
    def __init__(self, data_loader, model):
        super(Evaluator, self).__init__()
        self.data_loader = data_loader
        self.model = model

    def evaluate(self):
        self.model.eval()
    
        correct = 0
        total = 0
        for data in tqdm(self.data_loader):
            self.model.set_input(data)
            self.model.forward()
            labels = data[1]
            _, predicted = torch.max(self.model.logits.data, 1)
            total += labels.size(0)
            correct += (predicted == self.model.labels).sum()
        
        accuracy = int(correct)/int(total)
        return accuracy*100