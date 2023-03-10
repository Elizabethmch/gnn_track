import time
import torch
from torch import nn
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import torch.distributed as dist
from config import args
import math
import pandas as pd

def getLossAcc(criterion, output, labels, extra=None):
    # normalize labels, un-normalize output
    loss = criterion(output, labels).sum()

    # Acc
    _, predict = torch.max(output.detach(),1)
    acc = (predict==labels)

    return loss, acc


device = args.device
def train_one_epoch(model, trainloader, criterion, optimizer, res):
    model.train()
    
    timeBegin = time.time()
    total_n_data, total_correct, total_loss = torch.tensor(0.).to(device),\
                        torch.tensor(0.).to(device), torch.tensor(0.).to(device)
    for i, data in enumerate(trainloader, 0):
        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        # Model output
        data = data.to(device)
        target = data.y   
        output = model(data)
        
        loss, acc = getLossAcc(criterion, output, target, extra=data)    
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        optimizer.step()

        total_n_data += len(acc)
        total_correct += acc.sum().item()
        total_loss += loss.item()

    # record result
    if args.distributed:
        dist.barrier()
        for ary in [total_n_data, total_correct, total_loss]:   # collect data from all gpus
            dist.all_reduce(ary, op=dist.ReduceOp.SUM)      
    res['train_time'].append(time.time()-timeBegin)
    res['train_loss'].append(float(total_loss/total_n_data))
    res['train_acc'].append(float(total_correct/total_n_data))
    


def test_one_epoch(model, testloader, criterion, res, check_output = False):
    model.eval()
    timeBegin = time.time()
    # Record scores
    predLog = []

    total_n_data, total_correct, total_loss = 0, 0, 0
    for i, data in enumerate(testloader, 0):
        data = data.to(device)
        target = data.y 

        with torch.no_grad():
            output = model(data) 
            
            loss, acc = getLossAcc(criterion, output, target, extra=data)     

            total_n_data += len(acc)
            total_correct += acc.sum().item()
            total_loss += loss.item()
            predLog.append(torch.cat((target.view(-1,1), output), 1).detach().cpu().numpy() )

    if check_output:
        print(" labels: ", data.y[0:5])
        print(" output: ", output[0:5, :])
        print()

    # record result
    prediction = np.zeros((0,predLog[0].shape[1]))
    for s in predLog:
        prediction = np.concatenate( (prediction, s), 0)
    
    del predLog

    res['test_time'].append(time.time()-timeBegin)
    res['test_loss'].append(float(total_loss/total_n_data))
    res['test_acc'].append(float(total_correct/total_n_data))

    return prediction
