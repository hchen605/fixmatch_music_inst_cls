from tqdm import tqdm
import torch.nn.functional as F
from trainer.train_utils import *
import math,random


def trainer_mt(model, teacher, data_loader, optimizer, criterion, c_w, alpha, epoch_num):
    global_step = epoch_num * len(data_loader)
    model.train()
    teacher.train()
    class_loss_tracker = AverageMeter()
    consi_loss_tracker = AverageMeter()
    t_class_loss_tracker = AverageMeter()
    
    for i, (X, Y_true, Y_mask) in tqdm(enumerate(data_loader)):

        #print(' --- start mt ---')
        X = X.cuda()
        Y_true = Y_true.cuda()
        Y_mask = Y_mask.cuda()

        outputs = model(X)
        # print('outputted')
        
        # Regardless of what criterion or whether this is instrument-wise
        # Let the criterion function deal with it
        class_loss = criterion(outputs[Y_mask], Y_true[Y_mask])
        
        # Compute consistency loss here
        #tfmask
        #X = tfmask(X)
        outputs_ = teacher(X)
        consistency_loss = F.mse_loss(outputs, outputs_)
        
        loss = class_loss + c_w*consistency_loss

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        
        # Update teacher
        global_step += 1
        update_ema_variables(model, teacher, global_step, alpha)
        t_class_loss = criterion(outputs_[Y_mask], Y_true[Y_mask])
                                 
        # Update average meters
        class_loss_tracker.update(class_loss.item())
        consi_loss_tracker.update(consistency_loss.item())
        t_class_loss_tracker.update(t_class_loss.item())
    return (class_loss_tracker.avg, consi_loss_tracker.avg, t_class_loss_tracker.avg)



def tfmask(X):
    
    #(20000, 10, 128)
    tmask_length = random.randint(0, 3)
    tmask_start = random.randint(0, 7)
    X[:,tmask_start:tmask_start+tmask_length,:] = 0.
    
    fmask_length = random.randint(0, 25)
    fmask_start = random.randint(0, 90)
    X[:,:,fmask_start:fmask_start+fmask_length] = 0.
    
    return X