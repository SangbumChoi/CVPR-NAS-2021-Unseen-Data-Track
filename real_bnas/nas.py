import sys
import time
import numpy as np
import torch
import utils
import logging
import torch.nn as nn
import genotypes
import torch.utils
import torch.backends.cudnn as cudnn
import bin_utils_search

from _model import NetworkImageNet
from search import Network
from architect import Architect
from helpers import *

"""
Your submission must contain a class NAS that has a `search` method. The arguments
we'll pass and the output we'll expect are detailed below, but beyond that you can
do whatever you want within the class. Feel free to add other python files that are
imported, as long as you bundle them into the submission zip and they are imported when
`import nas` is called.

For information, you can check out the ingestion program to see exactly how we'll interact 
with this class. To submit, zip this file and any helpers files together with the dataset_metadata file
 and upload it to the CodaLab page.
"""

class NAS:
    def __init__(self):
        self.seed = 2
        self.gpu = 1
        self.batch_size = 64
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.weight_decay = 3e-4
        self.epochs = 10
        self.init_channels = 32
        self.layers = 6
        self.drop_path_prob = 0.2
        self.unrolled = False
        self.grad_clip = 5
        self.learning_rate_min = 0.001
        self.arch_learning_rate = 3e-4
        self.arch_weight_decay = 1e-3
        self.num_skip = 3
        self.label_smooth = 0.1
        self.image = False
        self.lamda = 1.0
        self.tau = 7.7
        self.gamma = 3
        self.geno_name = 'EXP'

        print('self : {}'.format(self))
    """
	search() Inputs:
		train_x: numpy array of shape (n_datapoints, channels, weight, height)
		train_y: numpy array of shape (n_datapoints)
		valid_x: numpy array of shape (n_datapoints, channels, weight, height)
		valid_y: numpy array of shape (n_datapoints)
		dataset_metadata: dict, contains:
			* batch_size: the batch size that will be used to train this da==taset
			* n_classes: the total number of classes in the classification task
			* lr: the learning rate that will be used to train this dataset
			* benchmark: the threshold used to determine zero point of scoring; your score on the dataset will equal
			 '10 * (test_acc - benchmark) / (100-benchmark)'. 
                - This means you can score a maximum of 10 points on each dataset: a full 10 points will be awarded for 100% test accuracy, 
                while 0 points will be awarded for a test accuracy equal to the benchmark. 
			* name: a unique name for this dataset
					
	search() Output:
		model: a valid PyTorch model
    """

    def search(self, train_x, train_y, valid_x, valid_y, metadata):
        
        np.random.seed(self.seed)
        # torch.cuda.set_device(self.gpu)
        cudnn.benchmark = True
        torch.manual_seed(self.seed)
        cudnn.enabled=True
        torch.cuda.manual_seed(self.seed)

        helper_function()
        n_classes = metadata['n_classes']
        
        # check torch available
        if not torch.cuda.is_available():
            logging.info('no gpu device available')
            sys.exit(1)

        cudnn.benchmark = True
        cudnn.enabled = True

        data_channel = np.array(train_x).shape[1]

        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()

        model = Network(self.init_channels, data_channel, n_classes, self.layers, criterion)
        model = model.cuda()

        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()

        optimizer = torch.optim.SGD(
            model.parameters(),
            self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

        architect = Architect(model, self)
        bin_op = bin_utils_search.BinOp(model, self)
        best_genotypes = []

        train_pack = list(zip(train_x, train_y))
        valid_pack = list(zip(valid_x, valid_y))

        train_loader = torch.utils.data.DataLoader(train_pack, int(self.batch_size), pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(valid_pack, int(self.batch_size), pin_memory=True)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(self.epochs), eta_min=self.learning_rate_min)

        best_accuracy = 0
        best_epoch = 0
        train_epoch = time.time()

        for epoch in range(self.epochs):
            print("=== SEARCH STAGE EPOCH {} ===".format(epoch))

            scheduler.step()
            lr = scheduler.get_last_lr()

            genotype = model.genotype()
            genotype_img = model.genotype(self.gamma)

            model.drop_path_prob = self.drop_path_prob * epoch / self.epochs

            train_acc, train_obj = train(train_loader, valid_loader, model, architect, criterion, optimizer, lr, bin_op, epoch)
            valid_acc, valid_obj = infer(valid_loader, model, criterion, bin_op)

            average_epoch_t = (time.time() - train_epoch) / (epoch + 1)

            if best_accuracy < valid_acc:
                best_accuracy = valid_acc
                best_epoch = epoch
                saved_model = model
                if len(best_genotypes) > 0:
                    best_genotypes[0] = genotype
                    best_genotypes[1] = genotype_img
                else:
                    best_genotypes.append(genotype)
                    best_genotypes.append(genotype_img)

            prog_str = "  Train Acc:  {:>8.3f}%, Val Acc: {:>8.3f}%, Mem Alloc: {}, T Remaining Est: {}".format(
                train_acc,
                valid_acc,
                cache_stats(human_readable=True),
                show_time(average_epoch_t * (self.epochs - epoch)))
            prog_str += "\n  Current best score:    Val Acc: {:>9.3f}% @ epoch {}".format(
                best_accuracy,
                best_epoch)
            prog_str += "\n genotype = {} ".format(best_genotypes)

            print(prog_str)

        with open('./genotypes.py', 'a') as f:
            f.write(self.geno_name + ' = ' + str(best_genotypes[0]) + '\n')
            f.write(self.geno_name + '_img' + ' = ' + str(best_genotypes[1]) + '\n')

        if data_channel == 3:
            print('==== data_channel is 3 ====')
            return_model = NetworkImageNet(self.init_channels, n_classes, self.layers, best_genotypes[0])
            return return_model
        return saved_model

def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, bin_op, epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()
    for step, (input, target) in enumerate(train_queue):
        bin_op.binarization()

        n = input.size(0)

        input =input.cuda()
        target =target.cuda()

        input_search, target_search = next(iter(valid_queue))
        input_search = input_search.cuda()
        target_search = target_search.cuda()

        architect.step(input, target, input_search, target_search, lr, optimizer,epoch)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        bin_op.restore()
        bin_op.updateBinaryGradWeight()
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

    return top1.avg, objs.avg

def infer(valid_queue, model, criterion, bin_op):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()
    bin_op.binarization()
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda()

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    bin_op.restore()
    return top1.avg, objs.avg
