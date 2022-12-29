# coding=utf-8

import sys
import datetime
import random
import numpy as np
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset as dataset
from apex import amp

from misc import *


def total_loss(pred, mask):
    pred = torch.sigmoid(pred)
    bce_loss = nn.BCELoss()
    bce = bce_loss(pred, mask)

    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter+1)/(union-inter+1)
    iou = iou.mean()
    return iou + 0.6*bce
    
def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

def bce_loss(pred, mask):
    pred = torch.sigmoid(pred)
    bce_loss = nn.BCELoss()
    bce = bce_loss(pred, mask)
    return bce


eps = 1e-6

def CEL(pred, target):
    pred = pred.sigmoid()
    intersection = pred * target
    numerator = (pred - intersection).sum() + (target - intersection).sum()
    denominator = pred.sum() + target.sum()
    return numerator / (denominator + eps)

def validate(model, val_loader, nums):
    model.train(False)
    avg_mae = 0.0
    with torch.no_grad():
        for image, mask, shape, name in val_loader:
            image, mask = image.cuda().float(), mask.cuda().float()
            out, _, _, _, _, _ = model(image)
            pred = torch.sigmoid(out[0, 0])
            #avg_mae += torch.abs(pred - mask[0]).mean()
            avg_mae += torch.mean(abs(pred - mask[0])).item()
            #pred = torch.sigmoid(out)
            #avg_mae += compute_mae(pred, mask[0])
    model.train(True)
    return (avg_mae / nums)
    

EXP_NAME = ''

def train(Dataset, Network, cfg):
    ## Set random seeds
    seed = 7
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    ## dataset
    data = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, num_workers=8)
    ## val dataloader
    val_cfg = Dataset.Config(datapath='/home/crh/MirrorDataset/MSD', mode='test')
    val_data = Dataset.Data(val_cfg)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=8)
    min_mae = 1.0
    best_epoch = 0
    ## network
    net = Network(cfg)
    net.train(True)
    net.cuda()
    ## parameter
    enc_params, dec_params = [], []
    for name, param in net.named_parameters():
        # print(name)
        if 'bkbone' in name:
            print('backbone: ', name)
            enc_params.append(param)
        else:
            dec_params.append(param)

    optimizer = torch.optim.SGD([{'params': enc_params}, {'params': dec_params}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)

    net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
    sw = SummaryWriter(cfg.savepath)
    global_step = 1

    # curr_iter = 1
    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr*0.1
        optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr

        for step, (image, mask, edge) in enumerate(loader):

            image, mask, edge = image.cuda().float(), mask.cuda().float(), edge.float().cuda()

            out1, out_edge1, out2,  out3, out4, out5 = net(image)
            loss1 = structure_loss(out1, mask)
            loss_edge = bce_loss(out_edge1, edge)
            loss2 = structure_loss(out2, mask)
  
            loss3 = structure_loss(out3, mask)

            loss4 = structure_loss(out4, mask)
            loss5 = structure_loss(out5, mask)
            loss = loss1 + loss_edge + loss2/2 + loss3/4 + loss4/8 + loss5/16

            optimizer.zero_grad()
            
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
            optimizer.step()

            

            ## log
            global_step += 1
            sw.add_scalar('lr', optimizer.param_groups[1]['lr'], global_step=global_step)
            sw.add_scalars('loss', {'loss1': loss1.item(), 'loss_edge1': loss_edge.item(), 'loss2': loss2.item(),
                                    'loss3': loss3.item(), 'loss4': loss4.item(),'loss5': loss5.item()}, global_step=global_step)
            if step % 10 == 0:
                print('%s | step:%d/%d/%d | lr=%.6f | loss=%.6f' % (datetime.datetime.now(), global_step, epoch+1, cfg.epoch, optimizer.param_groups[1]['lr'], loss.item()))


        if epoch+1 > cfg.epoch//2 or (epoch+1)%30 == 0:
            mae = validate(net, val_loader, 955) #571 955
            print('MSD MAE:%s' % mae)
            if mae < min_mae:
                min_mae = mae
                best_epoch = epoch + 1
                torch.save(net.state_dict(), cfg.savepath + '/model-best')
            print('best epoch is:%d, MAE:%s' % (best_epoch, min_mae))
            if epoch == 148 or epoch == 149:
                torch.save(net.state_dict(), cfg.savepath + '/model-' + str(epoch + 1))
        
        #scheduler.step()

if __name__ == '__main__':
    
    EXP_NAME = 'check-msd'
    from Net import Net
    cfg = dataset.Config(dataset='MSD', datapath='/home/crh/MirrorDataset/MSD', savepath=f'./{EXP_NAME}/', mode='train', batch=12, lr=0.01, momen=0.9, decay=5e-4, epoch=150)
    train(dataset, Net, cfg)
