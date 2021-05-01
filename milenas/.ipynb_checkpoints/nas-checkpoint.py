import argparse
import glob
import logging
import os
import sys
import time

import numpy as np
import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torchvision.datasets as dset

import utils
import genotypes

from torch.autograd import Variable
from model_search import Network
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
        self.batch_size = 256
        self.learning_rate = 0.025
        self.momentum = 0.9
        self.weight_decay = 3e-4
        self.epochs = 4
        self.init_channels = 8
        self.layers = 4
        self.drop_path_prob = 0.2
        self.unrolled = False
        self.grad_clip = 5
        self.learning_rate_min = 0.001
        self.report_freq = 10
        self.arch_learning_rate = 3e-4
        self.arch_weight_decay = 1e-3

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
        cudnn.benchmark = True
        torch.manual_seed(self.seed)
        cudnn.enabled=True
        torch.cuda.manual_seed(self.seed)
        is_multi_gpu = False
        
        helper_function()
        n_classes = metadata['n_classes']
        
        # check torch available
        if not torch.cuda.is_available():
            logging.info('no gpu device available')
            sys.exit(1)

        cudnn.benchmark = True
        cudnn.enabled = True
        
        # loading criterion
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
        
        train_pack = list(zip(train_x, train_y))
        valid_pack = list(zip(valid_x, valid_y))
        
        data_channel = np.array(train_x).shape[1]

        train_loader = torch.utils.data.DataLoader(train_pack, int(self.batch_size), pin_memory=True, num_workers=4)
        valid_loader = torch.utils.data.DataLoader(valid_pack, int(self.batch_size), pin_memory=True, num_workers=4)

        model = Network(self.init_channels, data_channel, n_classes, self.layers, criterion)
        model = model.cuda()

        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

        optimizer = torch.optim.SGD(
            model.parameters(),
            self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(self.epochs), eta_min=self.learning_rate_min)

        architect = Architect(model, criterion, self.momentum, self.weight_decay, self.arch_learning_rate, self.arch_weight_decay)

        best_accuracy = 0
        best_accuracy_different_cnn_counts = dict()
        
        
        for epoch in range(self.epochs):
            lr = scheduler.get_lr()[0]
            logging.info('epoch %d lr %e', epoch, lr)

            # training
            objs = utils.AvgrageMeter()
            top1 = utils.AvgrageMeter()
            top5 = utils.AvgrageMeter()

            train_batch = time.time()
            
            for step, (input, target) in enumerate(train_loader):
                
                # logging.info("epoch %d, step %d START" % (epoch, step))
                model.train()
                n = input.size(0)

                input = input.cuda()
                target = target.cuda()

                # get a random minibatch from the search queue with replacement
                input_search, target_search = next(iter(valid_loader))
                input_search = input_search.cuda()
                target_search = target_search.cuda()

                # Update architecture alpha by Adam-SGD
                # logging.info("step %d. update architecture by Adam. START" % step)
                # if args.optimization == "DARTS":
                #     architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
                # else:
                architect.step_milenas_2ndorder(input, target, input_search, target_search, lr, optimizer, 1, 1)

                # logging.info("step %d. update architecture by Adam. FINISH" % step)
                # Update weights w by SGD, ignore the weights that gained during architecture training

                # logging.info("step %d. update weight by SGD. START" % step)
                optimizer.zero_grad()
                logits = model(input)
                loss = criterion(logits, target)

                loss.backward()
                parameters = model.module.arch_parameters() if is_multi_gpu else model.arch_parameters()
                nn.utils.clip_grad_norm_(parameters, self.grad_clip)
                optimizer.step()

                # logging.info("step %d. update weight by SGD. FINISH\n" % step)

                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)

                # torch.cuda.empty_cache()

                if step % self.report_freq == 0:
                    average_batch_t = (time.time() - train_batch) / (step + 1)
                    print(" Step: {}, Top1: {}, Top5: {}, T: {}".format(step, top1.avg, top5.avg, show_time(average_batch_t * (len(train_loader) - step))))

            model.eval()
            
            # validation
            with torch.no_grad():
                    objs = utils.AvgrageMeter()
                    top1 = utils.AvgrageMeter()
                    top5 = utils.AvgrageMeter()

                    for step, (input, target) in enumerate(valid_loader):
                        input = input.cuda()
                        target = target.cuda()

                        logits = model(input)
                        loss = criterion(logits, target)

                        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                        n = input.size(0)
                        objs.update(loss.item(), n)
                        top1.update(prec1.item(), n)
                        top5.update(prec5.item(), n)

                        if step % self.report_freq == 0:
                            print(" Step: {}, Top1: {}, Top5: {}".format(step, top1.avg, top5.avg))
                                  
            scheduler.step()

            # save the structure
            genotype, normal_cnn_count, reduce_cnn_count = model.module.genotype() if is_multi_gpu else model.genotype()
            print("(n:%d,r:%d)" % (normal_cnn_count, reduce_cnn_count))
            print(F.softmax(model.module.alphas_normal if is_multi_gpu else model.alphas_normal, dim=-1))
            print(F.softmax(model.module.alphas_reduce if is_multi_gpu else model.alphas_reduce, dim=-1))
            logging.info('genotype = %s', genotype)
        
        return model