import argparse
import os, sys
import os.path as osp

def mkdirifmissing(path):
    if not osp.isdir(path):
        os.mkdir(path)

class Arguments(object):
    def __init__(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        working_dir = osp.dirname(osp.abspath(__file__))
        
        #### Experiment directory
        parser.add_argument('--exp-dir', type=str, metavar='PATH',default=osp.join(working_dir, './experiments'))
        
        #### Data options
        # Dataset
        parser.add_argument('--data-dir', type=str, metavar='PATH',default=osp.join(working_dir, './data'))
        # Dataloader
        parser.add_argument('--train-batch-size', type=int, default=16, help='train batch size')
        parser.add_argument('--test-batch-size', type=int, default=256, help='test batch size')
        parser.add_argument('--workers', default=4, type=int, help='num threads for loading data')
        parser.add_argument('--augmentations',default="",nargs='+', help='Augmentations to use on data')
        
        #### Model settings
        # Network parameters
        parser.add_argument('--base-ch-dim', type=int, default=32, help='Base conv output dimension, will increase at each step')
        parser.add_argument('--use-resblock', action='store_true', help='Use resisudal block at network structure')
        parser.add_argument('--use-batchnorm', action='store_true', help='Use batchnorm in convolution block')
        parser.add_argument('--conv-steps', type=int, default=2, help='How many conv blocks to use')
        parser.add_argument('--kernel-size', type=int, default=3, help='kernel size of conv filters')
        parser.add_argument('--dropout', type=float, default=0, help='Ratio of dropout before last fc layer, 0 means no dropout')

        #### Optimizer setting
        # Learning rate parameters
        parser.add_argument('--optimizer', type=str, default='adam', choices=['adam','sgd'],help='which optimizer to use')
        parser.add_argument('--lr', type=float, default=5e-4, help='initial learning rate')
        parser.add_argument('--epochs', type=int, default=50, help='How many epochs to run the model')
        # Scheduler parameters
        parser.add_argument('--lr-scheduler', type=str, default='step',choices=['step','expo'], help='Learning rate scheduler to use')

        #### Save/Eval Steps
        parser.add_argument('--save-step', type=int, default=3, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--eval-step', type=int, default=3, help='frequency of evaluate checkpoints at the end of epochs')
        
        self.args = parser.parse_args()
        ## Create missing directories
        mkdirifmissing(self.args.exp_dir)
        self.args.log_dir = osp.join(self.args.exp_dir,'train.log')
        self.args.ckpt_dir = osp.join(self.args.exp_dir,'ckpt')
        mkdirifmissing(self.args.ckpt_dir)
        
        self.show_args()

    def parse(self):
        return self.args

    def show_args(self):
        args = vars(self.args)
        fpath = osp.join(self.args.exp_dir,'arguments.txt')
        with open(fpath, 'w') as f:
            f.write('----------- Arguments ------------\n')
            for k, v in sorted(args.items()):
                msg = '%s: %s \n' % (str(k), str(v))
                f.write(msg)
                print(msg)
            f.write('-------------- End ---------------')