import os
import gc
import sys
import time
import glob
import numpy as np
import torchvision
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torch.backends.cudnn as cudnn
import bin_utils

from torch.autograd import Variable
from model import Network
from _model import EffNetV2
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
        self.gpu = 0
        self.arch = 'latest_cell_skip3'
        self.batch_size = 256
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.weight_decay = 3e-4
        self.epochs = 96
        self.init_channels = 32
        self.layers = 16
        self.drop_path_prob = 0.2
        self.unrolled = False
        self.grad_clip = 5
        self.learning_rate_min = 0.001
        self.report_freq = 20
        self.arch_learning_rate = 3e-4
        self.arch_weight_decay = 1e-3
        self.num_skip = 3
        self.label_smooth = 0.1
        self.image = False

        print('arch:{}\nbatch_size:{}\nlearning_rate:{}\nmomentum:{}\nweight_deacy:{}\nepochs:{}\ninit_channels:{}\nlayers:{}\ndrop_path_prob:{}\nunrolled:{}'.format(self.arch, 
            self.batch_size,
            self.learning_rate,
            self.momentum,
            self.weight_decay,
            self.epochs,
            self.init_channels,
            self.layers,
            self.drop_path_prob,
            self.unrolled))
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
        torch.cuda.set_device(self.gpu)
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

        data_channel = np.array(train_x).shape[1]

        genotype = eval("genotypes.%s" % self.arch)
        if data_channel==3:
            print("=== SEARCH STAGE DATA_CHANNEL {} ===".format(data_channel))

            # load resnet18 model
            model = torchvision.models.resnext101_32x8d()

            # reshape it to this dataset
            model.conv1 = nn.Conv2d(train_x.shape[1], 64, kernel_size=(7, 7), stride=1, padding=3)
            model.fc = nn.Linear(model.fc.in_features, n_classes, bias=True)
            return model
        model = Network(self.init_channels, data_channel ,n_classes, self.layers, genotype)
        model.cuda()
        
        # loading criterion and optimizer
        bin_op = bin_utils.BinOp(model, self)

        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()

        # optimizer = torch.optim.SGD(
        #     model.parameters(),
        #     self.learning_rate,
        #     momentum=self.momentum,
        #     weight_decay=self.weight_decay
        # )
       
        optimizer = torch.optim.Adam(model.parameters(), self.learning_rate, weight_decay=self.weight_decay)


        train_pack = list(zip(train_x, train_y))
        valid_pack = list(zip(valid_x, valid_y))

        train_loader = torch.utils.data.DataLoader(train_pack, int(self.batch_size), pin_memory=True, num_workers=4)
        valid_loader = torch.utils.data.DataLoader(valid_pack, int(self.batch_size), pin_memory=True, num_workers=4)

        # since submission server does not deal with multi-gpu
        if is_multi_gpu:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 50, eta_min = 0.0) if self.image else utils.OneCycleLR(optimizer, self.epochs * len(train_loader), lr_range=(0.05 * self.learning_rate, self.learning_rate))

        best_accuracy = 0
        best_epoch = 0
        train_epoch = time.time()

        for epoch in range(self.epochs):
            print("=== SEARCH STAGE EPOCH {} ===".format(epoch))

            model.drop_path_prob = self.drop_path_prob * epoch / self.epochs

            train_acc, train_obj = _train(train_loader, model, criterion, optimizer, bin_op, scheduler, epoch, self.grad_clip, self.report_freq) if self.image else train(train_loader, model, criterion, optimizer, bin_op, scheduler, self.report_freq)
            valid_acc, valid_obj = infer(valid_loader, model, criterion, bin_op, self.report_freq)

            average_epoch_t = (time.time() - train_epoch) / (epoch + 1)

            if best_accuracy < valid_acc:
                best_accuracy = valid_acc
                best_epoch = epoch
                saved_model = model

            prog_str = "  Train Acc:  {:>8.3f}%, Val Acc: {:>8.3f}%, Mem Alloc: {}, T Remaining Est: {}".format(
                train_acc,
                valid_acc,
                cache_stats(human_readable=True),
                show_time(average_epoch_t * (self.epochs - epoch)))
            prog_str += "\n  Current best score:    Val Acc: {:>9.3f}% @ epoch {}".format(
                best_accuracy,
                best_epoch)
            print(prog_str)

        return saved_model

def train(train_queue, model, criterion, optimizer, bin_op, scheduler, report_freq):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        bin_op.binarization()
        input = input.cuda()
        target = target.cuda()

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)
        loss.backward()
        bin_op.restore()
        bin_op.updateBinaryGradWeight()
        optimizer.step()
        scheduler.step()
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # if step % report_freq == 0:
        #     print("Step: {}, Top1: {}, Top5: {}".format(step, top1.avg, top5.avg))

    return top1.avg, objs.avg

def _train(train_queue, model, criterion, optimizer, bin_op, scheduler, epoch, grad_clip, report_freq):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()
    iters = len(train_queue)
    for step, (input, target) in enumerate(train_queue):

        bin_op.binarization()
        target = target.cuda()
        input = input.cuda()
        scheduler.step(epoch + step/iters)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), grad_clip)
        bin_op.restore()
        bin_op.updateBinaryGradWeight()
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        # reduced_loss = loss.data
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        # objs.update(to_python_float(reduced_loss), n)
        # top1.update(to_python_float(prec1), n)
        # top5.update(to_python_float(prec5), n)
        # if step % report_freq == 0:
        #     print("Step: {}, Top1: {}, Top5: {}".format(step, top1.avg, top5.avg))
        del loss, logits, input, target, prec1, prec5
        gc.collect()
        torch.cuda.empty_cache()
    return top1.avg, objs.avg

def infer(valid_queue, model, criterion, bin_op, report_freq):
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

            # if step % report_freq == 0:
            #     print("Step: {}, Top1: {}, Top5: {}".format(step, top1.avg, top5.avg))
    bin_op.restore()
    return top1.avg, objs.avg
