from tqdm import tqdm
import torch.nn.functional as F
from trainer.train_utils import *
import math,random
import numpy as np


def trainer_fixmatch(model, teacher, data_loader, optimizer, criterion, c_w, alpha, epoch_num, file_mask):
    global_step = epoch_num * len(data_loader)
    model.train()
    teacher.train()
    class_loss_tracker = AverageMeter()
    consi_loss_tracker = AverageMeter()
    t_class_loss_tracker = AverageMeter()
    pos_th = 0.95
    neg_th = 0.05
    
    #file_mask = open("mask.txt", "w") 
    mask_sum = 0
    #file_con_loss = open("con_loss.txt", "w") 
    #file_label_loss_s = open("label_loss_s.txt", "w")
    #file_label_loss_s = open("label_loss_s.txt", "w")
    
    
    for i, (X, Y_true, Y_mask) in tqdm(enumerate(data_loader)):

        #print(' --- start mt ---')
        X = X.cuda()
        Y_true = Y_true.cuda()
        Y_mask = Y_mask.cuda()
        #print(Y_mask.shape)#64,20

        #weakly aug
        X_aug = tfmask_weak(X, 2, 10)
        outputs = model(X_aug)
        # print('outputted')
        
        # Regardless of what criterion or whether this is instrument-wise
        # Let the criterion function deal with it
        # labeled loss
        class_loss = criterion(outputs[Y_mask], Y_true[Y_mask])
        
        # calculate pseudo label for fixmatch
        mask_pos = outputs > pos_th
        mask_neg = outputs < neg_th
        mask_all = mask_pos ^ mask_neg
        #print(mask_all)
        
        mask_sum = mask_sum + np.sum(mask_all.cpu().numpy())
        
        #print(mask_all.shape) #64,20
        #if i == 0:
        #    print('mask_all')
        #    print(mask_all[1])
        
        outputs_pseudo = binarize_targets_pos(outputs, pos_th)
        outputs_pseudo = binarize_targets_neg(outputs_pseudo, neg_th)
        # Compute consistency loss here
        #strongly aug
        #tfmask
        X_aug = tfmask_strong(X, 3, 30)
        outputs_ = teacher(X_aug)
        #outputs = binarize_targets(outputs, 0.7)
        
        consistency_loss = criterion(outputs_pseudo[mask_all], outputs_[mask_all])
        
        #train_op = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(loss_xe + wu * loss_xeu + wd * loss_wd, colocate_gradients_with_ops=True)
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
    
    
    print('--- Mask Sum = ', mask_sum, '----\n')
    file_mask.writelines(str(mask_sum)+',')
    
    return (class_loss_tracker.avg, consi_loss_tracker.avg, t_class_loss_tracker.avg)



def tfmask_weak(X, t, f):
    
    #(20000, 10, 128)
    tmask_length = random.randint(1, t)
    tmask_start = random.randint(0, 7)
    #X[:,tmask_start:tmask_start+tmask_length,:] = 0.
    
    fmask_length = random.randint(2, f)
    fmask_start = random.randint(0, 70)
    X[:,:,fmask_start:fmask_start+fmask_length] = 0.
    
    return X

def tfmask_strong(X, t, f):
    
    #(20000, 10, 128)
    tmask_length = random.randint(2, t)
    tmask_start = random.randint(0, 7)
    #X[:,tmask_start:tmask_start+tmask_length,:] = 0.
    
    fmask_length = random.randint(20, f)
    fmask_start = random.randint(0, 70)
    X[:,:,fmask_start:fmask_start+fmask_length] = 0.
    
    return X

def binarize_targets(targets, threshold=0.5):
    targets[targets < threshold] = 0
    targets[targets > 0] = 1
    return targets

def binarize_targets_pos(targets, threshold=0.5):
    #targets[targets < threshold] = 0
    targets[targets > threshold] = 1
    return targets

def binarize_targets_neg(targets, threshold=0.5):
    targets[targets < threshold] = 0
    #targets[targets > 0] = 1
    return targets