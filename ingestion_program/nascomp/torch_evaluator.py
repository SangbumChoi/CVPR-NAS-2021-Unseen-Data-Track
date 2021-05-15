import numpy as np
import random
import time
import warnings


import torch
import torch.nn as nn
import torch.optim as optim

try:
    from nascomp.helpers import *
except ModuleNotFoundError:
    from ingestion_program.nascomp.helpers import *

import torch
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


# === TOP-K ACCURACY CALCULATION =======================================================================================
def top_k_accuracy(output, target, top_k):
    if len(output.shape) == 2:
        output = output.reshape(list(output.shape) + [1, 1])
        target = target.reshape(list(target.shape) + [1, 1])
    correct = np.zeros(len(top_k))
    _, pred = output.topk(max(top_k), 1, True, True)
    for i, k in enumerate(top_k):
        target_expand = target.unsqueeze(1).repeat(1, k, 1, 1)
        equal = torch.max(pred[:, :k, :, :].eq(target_expand), 1)[0]
        correct[i] = torch.sum(equal)
    return correct, len(target.view(-1)), equal.cpu().numpy()


# === MODEL UTILITIES ==================================================================================================
def general_num_params(m):
    # return number of differential parameters of input model
    return sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, m.parameters())])


def reset_weights(model):
    warn_non_resets = []
    diff_non_resets = []
    for module in model.modules():
        if type(module) != type(model):
            if 'reset_parameters' in dir(module):
                module.reset_parameters()
            else:
                if 'parameters' in dir(module):
                    n_params = general_num_params(module)
                    child_params = sum([general_num_params(m) for m in module.children()])

                    if n_params != 0 and n_params != child_params:
                        diff_non_resets.append([type(module).__name__, n_params])
                else:
                    warn_non_resets.append(type(module).__name__)

    if len(diff_non_resets):
        error_string = "\n".join(["\t* {}: {:,} parameter(s)".format(m, p) for m, p in diff_non_resets])
        raise AttributeError(
            "{} module(s) have differentiable parameters without a 'reset_parameters' function: \n {}".format(
                len(diff_non_resets),
                error_string))
    if len(warn_non_resets):
        warning_msg = "Model contains modules without 'reset_parameters' attribute: "
        warnings.warn(warning_msg + str(set(warn_non_resets)))


def sizeof_fmt(num, spacing=True, suffix='B'):
    # turns bytes object into human readable
    if spacing:
        fmt = "{:>7.2f}{:<3}" 
    else:
        fmt = "{:.2f}{}" 
        
    for unit in ['', 'Ki','Mi']:
        if abs(num) < 1024.0:
            return fmt.format(num, unit+suffix)
        num /= 1024.0
    return fmt.format(num, 'Gi'+suffix)


def cache_stats(human_readable=True, spacing=True):
    if not torch.cuda.is_available():
        return 0
    # returns current allocated torch memory
    if human_readable:
        return sizeof_fmt(torch.cuda.memory_reserved(), spacing)
    else:
        return int(torch.cuda.memory_reserved())


# === TRAIN/TEST FUNCTIONS =============================================================================================
def train(model, device, optimizer, criterion, train_loader, full_train):
    # define some tracking stats
    corrects, divisor, cumulative_loss = 0, 0, 0

    # === train epoch =========================
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        # pass data ===========================
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model.forward(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # compile batch results
        cumulative_loss += loss.item()
        corr, div, _ = top_k_accuracy(output, target, top_k=[1])
        corrects = corrects + corr
        divisor += div

        if not full_train and batch_idx>2:
            break

    acc = 100. * corrects / float(divisor)
    return cumulative_loss/len(train_loader), acc


def evaluate(model, device, criterion, loader):
    # === tracking stats =====================
    corrects, divisor, cumulative_loss = 0, 0, 0
    accuracy_dict = {}

    # === test epoch =========================
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            output = model.forward(data)
            loss = criterion(output, target)

            cumulative_loss += loss.item()
            corr, div, equal = top_k_accuracy(output, target, top_k=[1])
            corrects += corr
            divisor += div
            for i in range(len(equal)):
                accuracy_dict["{}_{}".format(batch_idx, i)] = equal[i]
            acc = 100 * corrects / float(divisor)
    return cumulative_loss/len(loader), acc, accuracy_dict


def predict(model, device, loader):
    # === tracking stats =====================
    predictions = []

    # === test epoch =========================
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data = data.to(device)
            output = model.forward(data)
            predictions += torch.argmax(output, 1).detach().cpu().tolist()
    return np.array(predictions)


def bootstrap_accuracy(accuracy_dict, n=1000):
    accuracies = []
    values = np.array(list(accuracy_dict.values()))[:, 0, 0]
    n_values = len(values)
    for _ in range(n):
        accuracies.append(np.random.choice(values, n_values, replace=True).sum()/n_values)
    return np.quantile(accuracies, q=[.025, .975])


# === FULL N EPOCH TRAIN =============================================================================================
def full_training(model, **kwargs):
    # grab necessary kwargs
    device = kwargs['device']
    epochs = kwargs['epochs']
    lr = kwargs['lr']
    train_loader = kwargs['train_loader']
    valid_loader = kwargs['valid_loader']
    test_loader = kwargs['test_loader']
    full_train = kwargs['full_train']
    if not full_train:
        epochs = 1

    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=.9, weight_decay=3e-4)

    # sum reduction to match tensorflow
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0
    best_epoch = 0
    best_test = None

    train_results, valid_results, test_results = [], [], []

    train_start = time.time()
    for epoch in range(epochs):
        print("=== EPOCH {} ===".format(epoch))
        train_loss, train_acc = train(model, device, optimizer, criterion, train_loader, full_train)
        valid_loss, valid_acc, valid_acc_dict = evaluate(model, device, criterion, valid_loader)
        train_results.append(train_acc[0])
        valid_results.append(valid_acc[0])
        test_predictions = predict(model, device, test_loader)

        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            best_test = test_predictions
            best_epoch = epoch
        scheduler.step()

        average_epoch_t = (time.time() - train_start) / (epoch + 1)
        prog_str = "  Train Acc:  {:>8.3f}%, Val Acc: {:>8.3f}%, Mem Alloc: {}, T Remaining Est: {}".format(
            train_acc[0],
            valid_acc[0],
            cache_stats(human_readable=True),
            show_time(average_epoch_t * (epochs - epoch)))
        prog_str += "\n  Train Loss: {:>8.3f} , Val Loss: {:>8.3f}".format(
            train_loss,
            valid_loss)
        prog_str += "\n  Current best score:    Val Acc: {:>9.3f}% @ epoch {}".format(
            best_val_acc[0],
            best_epoch)
        print(prog_str)

    best_epoch = int(np.argmax(valid_results))
    return {'train_accuracies': train_results, 
            'valid_accuracies': valid_results, 
            'test_predictions': best_test,
            'best_epoch': best_epoch,
            'best_val_score': best_val_acc[0]}


# === MAIN EVALUATION FUNCTION =========================================================================================
def torch_evaluator(model, data, metadata, n_epochs, full_train, verbose=False):
    print("===== EVALUATING {} =====".format(metadata['name']))
    print("Cuda available?", torch.cuda.is_available())

    # load + package data
    (train_x, train_y), (valid_x, valid_y), test_x = data
    batch_size = metadata['batch_size']
    lr = metadata['lr']

    train_pack = list(zip(train_x, train_y))
    valid_pack = list(zip(valid_x, valid_y))

    train_loader = torch.utils.data.DataLoader(train_pack, int(batch_size), shuffle=False)
    valid_loader = torch.utils.data.DataLoader(valid_pack, int(batch_size))
    test_loader = torch.utils.data.DataLoader(test_x, int(batch_size))

    # load and reset model
    reset_weights(model)

    # train
    results = full_training(
        model=model,
        device=torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu"),
        lr=lr,
        epochs=n_epochs,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        full_train=full_train,
        verbose=verbose
    )

    return results
