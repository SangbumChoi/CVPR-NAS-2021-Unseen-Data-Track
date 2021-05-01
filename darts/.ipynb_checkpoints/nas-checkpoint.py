import genotypes
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch
import torchvision
import utils
import logging
import torch.nn.functional as F
import sys

from torch.autograd import Variable
from model_search import Network
from architect import Architect

import helpers

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
        self.batch_size = 128
        self.learning_rate = 0.025
        self.momentum = 0.9
        self.weight_decay = 3e-4
        self.epochs = 1
        self.init_channels = 6
        self.layers = 4
        self.drop_path_prob = 0.2
        self.unrolled = False
        self.grad_clip = 5
        self.learning_rate_min = 0.001
        self.report_freq = 50

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
        
        helpers.helper_function()
        n_classes = metadata['n_classes']
        
#         reshape it to this dataset
#         model = torchvision.models.resnet18()
#         model.conv1 = nn.Conv2d(train_x.shape[1], 64, kernel_size=(7, 7), stride=1, padding=3)
#         model.fc = nn.Linear(model.fc.in_features, n_classes, bias=True)
#         return model

        if not torch.cuda.is_available():
            logging.info('no gpu device available')
            sys.exit(1)

        cudnn.benchmark = True
        cudnn.enabled = True

        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()

        model = Network(self.init_channels, n_classes, self.layers, criterion)
        model = model.cuda()

        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

        optimizer = torch.optim.SGD(
            model.parameters(),
            self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(self.epochs), eta_min=self.learning_rate_min)

        architect = Architect(model)

        train_pack = list(zip(train_x, train_y))
        valid_pack = list(zip(valid_x, valid_y))

        train_loader = torch.utils.data.DataLoader(train_pack, int(self.batch_size), shuffle=False)
        valid_loader = torch.utils.data.DataLoader(valid_pack, int(self.batch_size))

        for epoch in range(self.epochs):
            scheduler.step()
            lr = scheduler.get_lr()[0]
            logging.info('epoch %d lr %e', epoch, lr)

            genotype = model.genotype()
            logging.info('genotype = %s', genotype)

#             print(F.softmax(model.alphas_normal, dim=-1))
#             print(F.softmax(model.alphas_reduce, dim=-1))
            
            # training
            print("++++++Start training+++++++")
            for step, (input, target) in enumerate(train_loader):
                model.train()
                n = input.size(0)

                input = Variable(input, requires_grad=False).cuda()
                target = Variable(target, requires_grad=False).cuda(non_blocking=True)

                # get a random minibatch from the search queue with replacement
                input_search, target_search = next(iter(valid_loader))
                input_search = Variable(input_search, requires_grad=False).cuda()
                target_search = Variable(target_search, requires_grad=False).cuda(non_blocking=True)

                architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=self.unrolled)

                optimizer.zero_grad()
                logits = model(input)
                loss = criterion(logits, target)

                loss.backward()
                nn.utils.clip_grad_norm(model.parameters(), self.grad_clip)
                optimizer.step()
                
                if step % self.report_freq == 0:
                    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                    print(step, loss, prec1, prec5)

            # validation
            print("++++++Start validation+++++++")
            with torch.no_grad():
                for step, (input, target) in enumerate(valid_loader):
                    input = Variable(input).cuda()
                    target = Variable(target).cuda(non_blocking=True)

                    model.eval()
                    
                    logits = model(input)
                    loss = criterion(logits, target)

                    if step % self.report_freq == 0:
                        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                        print(step, loss, prec1, prec5)

        return model