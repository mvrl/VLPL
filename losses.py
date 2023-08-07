import torch

'''
loss functions
'''
def loss_bce(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    # input validation:
    assert not torch.any(observed_labels == -1)
    assert P['train_set_variant'] == 'clean'
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = neg_log(1.0 - preds[observed_labels == 0])
    reg_loss = None
    return loss_mtx, reg_loss

def loss_bce_ls(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    # input validation: 
    assert not torch.any(observed_labels == -1)
    assert P['train_set_variant'] == 'clean'
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = (1.0 - P['ls_coef']) * neg_log(preds[observed_labels == 1]) + P['ls_coef'] * neg_log(1.0 - preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = (1.0 - P['ls_coef']) * neg_log(1.0 - preds[observed_labels == 0]) + P['ls_coef'] * neg_log(preds[observed_labels == 0])
    reg_loss = None
    return loss_mtx, reg_loss


def loss_iun(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    true_labels = batch['label_vec_true']
    # input validation:
    assert torch.min(observed_labels) >= 0
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[true_labels == 0] = neg_log(1.0 - preds[true_labels == 0])  # FIXME
    reg_loss = None
    return loss_mtx, reg_loss

def loss_iu(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    # input validation: 
    assert torch.any(observed_labels == 1) # must have at least one observed positive
    assert torch.any(observed_labels == -1) # must have at least one observed negative
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == -1] = neg_log(1.0 - preds[observed_labels == -1])
    reg_loss = None
    return loss_mtx, reg_loss

def loss_pr(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    batch_size = int(batch['label_vec_obs'].size(0))
    num_classes = int(batch['label_vec_obs'].size(1))
    # input validation: 
    assert torch.min(observed_labels) >= 0
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    for n in range(batch_size):
        preds_neg = preds[n, :][observed_labels[n, :] == 0]
        for i in range(num_classes):
            if observed_labels[n, i] == 1:
                torch.nonzero(observed_labels[n, :])
                loss_mtx[n, i] = torch.sum(torch.clamp(1.0 - preds[n, i] + preds_neg, min=0))
    reg_loss = None
    return loss_mtx, reg_loss

def loss_an(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    true_labels = batch['label_vec_true'].to(Z['device'])
    # input validation:
    assert torch.min(observed_labels) >= 0
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = neg_log(1.0 - preds[observed_labels == 0])
    reg_loss = None

    return loss_mtx, reg_loss

def loss_an_ls(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    # input validation: 
    assert torch.min(observed_labels) >= 0
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = (1.0 - P['ls_coef']) * neg_log(preds[observed_labels == 1]) + P['ls_coef'] * neg_log(1.0 - preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = (1.0 - P['ls_coef']) * neg_log(1.0 - preds[observed_labels == 0]) + P['ls_coef'] * neg_log(preds[observed_labels == 0])
    reg_loss = None
    return loss_mtx, reg_loss

def loss_wan(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    # input validation: 
    assert torch.min(observed_labels) >= 0
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = neg_log(1.0 - preds[observed_labels == 0]) / float(P['num_classes'] - 1)
    reg_loss = None
    
    return loss_mtx, reg_loss

def loss_epr(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    # input validation:
    assert torch.min(observed_labels) >= 0
    # compute loss w.r.t. observed positives:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    # compute regularizer: 
    reg_loss = expected_positive_regularizer(preds, P['expected_num_pos'], norm='2') / (P['num_classes'] ** 2)
    return loss_mtx, reg_loss

def loss_role(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    estimated_labels = batch['label_vec_est']
    # input validation:
    assert torch.min(observed_labels) >= 0
    # (image classifier) compute loss w.r.t. observed positives:
    loss_mtx_pos_1 = torch.zeros_like(observed_labels)
    loss_mtx_pos_1[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    # (image classifier) compute loss w.r.t. label estimator outputs:
    estimated_labels_detached = estimated_labels.detach()
    loss_mtx_cross_1 = estimated_labels_detached * neg_log(preds) + (1.0 - estimated_labels_detached) * neg_log(1.0 - preds)
    # (image classifier) compute regularizer: 
    reg_1 = expected_positive_regularizer(preds, P['expected_num_pos'], norm='2') / (P['num_classes'] ** 2)
    # (label estimator) compute loss w.r.t. observed positives:
    loss_mtx_pos_2 = torch.zeros_like(observed_labels)
    loss_mtx_pos_2[observed_labels == 1] = neg_log(estimated_labels[observed_labels == 1])
    # (label estimator) compute loss w.r.t. image classifier outputs:
    preds_detached = preds.detach()
    loss_mtx_cross_2 = preds_detached * neg_log(estimated_labels) + (1.0 - preds_detached) * neg_log(1.0 - estimated_labels)
    # (label estimator) compute regularizer:
    reg_2 = expected_positive_regularizer(estimated_labels, P['expected_num_pos'], norm='2') / (P['num_classes'] ** 2)
    # compute final loss matrix:
    reg_loss = 0.5 * (reg_1 + reg_2)
    loss_mtx = 0.5 * (loss_mtx_pos_1 + loss_mtx_pos_2)
    loss_mtx += 0.5 * (loss_mtx_cross_1 + loss_mtx_cross_2)
    
    return loss_mtx, reg_loss


def loss_EM(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    true_labels = batch['label_vec_true'].to(Z['device'])

    # input validation:
    assert torch.min(observed_labels) >= 0

    loss_mtx = torch.zeros_like(preds)

    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = -P['alpha'] * (
            preds[observed_labels == 0] * neg_log(preds[observed_labels == 0]) +
            (1 - preds[observed_labels == 0]) * neg_log(1 - preds[observed_labels == 0])
        )

    return loss_mtx, None

def loss_EM_PL(batch, P, Z):
    # unpack:
    preds = batch['preds']
    #print("preds: ", preds)
    observed_labels = batch['label_vec_obs']
    #print("observed_labels: ", observed_labels)
    true_labels = batch['label_vec_true']
    #print("true_labels: ", true_labels)
    gamma_neg = 4
    gamma_pos = 2
    clip = 0.05
    
    if P['epoch'] > P['warmup_epoch']:
        pseudo_labels = batch['logits_pl']
        #print("pseudo_labels: ", pseudo_labels)
        similarity = batch['similarity']
        #print("similarity: ", similarity)
        final_labels = torch.where(observed_labels == 0, pseudo_labels, observed_labels)
        #print("final_labels: ", final_labels)
        #input validation:
        assert torch.min(final_labels) >= -1
        
        loss_mtx = torch.zeros_like(preds)
        #####
        #observed positive label
        #####
        loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1]) #+ neg_log(similarity[observed_labels == 1])
        #loss_mtx[final_labels == 1] = neg_log(preds[final_labels == 1]) #+ neg_log(similarity[final_labels == 1])
        
        #####
        #Unknown Labels
        #####
        loss_mtx[final_labels == 0] = -P['alpha'] * (preds[final_labels == 0] * neg_log(preds[final_labels == 0]) + (1 - preds[final_labels == 0]) * neg_log(1 - preds[final_labels == 0]))#-P['alpha'] *(similarity[final_labels == 0] * neg_log(similarity[final_labels == 0]) + (1 - similarity[final_labels == 0]) * neg_log(1 - similarity[final_labels == 0]))
   
        #####
        #Pseudo-Label
        #####
        #positive pseudo-label 
        mask_pos = (observed_labels  == 0) & (pseudo_labels == 1)
        #print(mask)
        loss_mtx[mask_pos] = P['beta_pos'] * (0.9* neg_log(preds[mask_pos]) + 0.1* neg_log(1-preds[mask_pos]))
        #loss_mtx[pseudo_labels==1] = P['beta_pos'] * neg_log(preds[pseudo_labels==1]) #+ P['beta_pos'] * neg_log(similarity[pseudo_labels==1]))
        
        '''
        #negative pseudo-label  
        #may also need to introduce the similarity_score to make sure train the model has the ability to discover the labels which         100% sure negative.  
        #negative pseudo-label 
        mask_neg = (observed_labels  == 0) & (pseudo_labels ==-1)
        loss_mtx[mask_neg] = P['beta_neg'] *(0.1 * neg_log(preds[mask_neg]) + 0.9 * neg_log(1 - preds[mask_neg]))
        #loss_mtx[mask_neg] = P['beta_neg'] * neg_log(1 - preds[mask_neg])# - P['beta_neg'] * neg_log(similarity[mask_neg]))
        '''
    else:
        #Using EM loss to warmup the whole model
        #print("The warmup starting...")
        loss_mtx = torch.zeros_like(preds)
        loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
        loss_mtx[observed_labels == 0] = -P['alpha'] * (
        preds[observed_labels == 0] * neg_log(preds[observed_labels == 0]) +
            (1 - preds[observed_labels == 0]) * neg_log(1 - preds[observed_labels == 0])
        )
    return loss_mtx, None


def loss_EM_PL_ASL(batch, P, Z):
    # unpack:
    preds = batch['preds']
    preds_neg = 1 - preds
    #print("preds: ", preds)
    observed_labels = batch['label_vec_obs']
    #print("observed_labels: ", observed_labels)
    true_labels = batch['label_vec_true']
    #print("true_labels: ", true_labels)
    gamma_neg = P['negative']
    gamma_pos = P['positive']
    gamma_unknown = P['unknown']
    clip = 0.05
    
    if clip is not None and clip > 0:
            preds_neg = (preds_neg + clip).clamp(max=1)
    
    if P['epoch'] > P['warmup_epoch']:
        pseudo_labels = batch['logits_pl']
        #print("pseudo_labels: ", pseudo_labels)
        similarity = batch['similarity']
        #print("similarity: ", similarity)
        final_labels = torch.where(observed_labels == 0, pseudo_labels, observed_labels)
        #print("final_labels: ", final_labels)
        #input validation:
        assert torch.min(final_labels) >= -1
        
        loss_mtx = torch.zeros_like(preds)
        pt = torch.zeros_like(preds)
        #####
        #observed positive label
        #####
        pt[observed_labels == 1] = preds[observed_labels == 1]
        
        loss_mtx[observed_labels == 1] = torch.pow(1 - pt[observed_labels == 1], gamma_pos) * neg_log(preds[observed_labels == 1]) #+ neg_log(similarity[observed_labels == 1])
        #loss_mtx[final_labels == 1] = neg_log(preds[final_labels == 1]) #+ neg_log(similarity[final_labels == 1])
        
        #####
        #Unknown Labels
        #####
        pt[final_labels == 0] = preds[final_labels == 0]
        
        loss_mtx[final_labels == 0] = - torch.pow(1 - pt[final_labels == 0], gamma_unknown) * (preds[final_labels == 0] * neg_log(preds[final_labels == 0]) + (1 - preds[final_labels == 0]) * neg_log(1 - preds[final_labels == 0])) #-P['alpha'] *(similarity[final_labels == 0] * neg_log(similarity[final_labels == 0]) + (1 - similarity[final_labels == 0]) * neg_log(1 - similarity[final_labels == 0]))
        #####
        #Pseudo-Label
        #####
        
        #positive pseudo-label 
        mask_pos = (observed_labels  == 0) & (pseudo_labels == 1)
        #print(mask)
        pt[mask_pos] = preds[mask_pos]
        loss_mtx[mask_pos] = torch.pow(1 - pt[mask_pos], gamma_pos) * neg_log(preds[mask_pos])
        #loss_mtx[pseudo_labels==1] = P['beta_pos'] * neg_log(preds[pseudo_labels==1]) #+ P['beta_pos'] * neg_log(similarity[pseudo_labels==1]))
        
        #negative pseudo-label  
        #may also need to introduce the similarity_score to make sure train the model has the ability to discover the labels which         100% sure negative.  
        #negative pseudo-label 
        mask_neg = (observed_labels  == 0) & (pseudo_labels ==-1)
        pt[mask_neg] = 1 - preds[mask_neg]
        loss_mtx[mask_neg] = torch.pow(1 - pt[mask_neg], gamma_neg) *(neg_log(1 - preds[mask_neg]))
        #loss_mtx[mask_neg] = P['beta_neg'] * neg_log(1 - preds[mask_neg])# - P['beta_neg'] * neg_log(similarity[mask_neg]))

    else:
        #Using EM loss to warmup the whole model
        #print("The warmup starting...")
        loss_mtx = torch.zeros_like(preds)
        loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
        loss_mtx[observed_labels == 0] = -P['alpha'] * (
        preds[observed_labels == 0] * neg_log(preds[observed_labels == 0]) +
            (1 - preds[observed_labels == 0]) * neg_log(1 - preds[observed_labels == 0])
        )
    return loss_mtx, None



def loss_EM_APL(batch, P, Z):
    # unpack:
    preds = batch['preds']
    #print("preds: ", preds)
    observed_labels = batch['label_vec_obs']
    #print("observed_labels: ", observed_labels)
    # input validation:
    assert torch.min(observed_labels) >= -1

    loss_mtx = torch.zeros_like(preds)

    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = -P['alpha'] * (
            preds[observed_labels == 0] * neg_log(preds[observed_labels == 0]) +
            (1 - preds[observed_labels == 0]) * neg_log(1 - preds[observed_labels == 0])
        )

    soft_label = -observed_labels[observed_labels < 0]
    loss_mtx[observed_labels < 0] = P['beta'] * (
            soft_label * neg_log(preds[observed_labels < 0]) +
            (1 - soft_label) * neg_log(1 - preds[observed_labels < 0])
        )
    return loss_mtx, None


loss_functions = {
    'bce': loss_bce,
    'bce_ls': loss_bce_ls,
    'iun': loss_iun,
    'iu': loss_iu,
    'pr': loss_pr,
    'an': loss_an,
    'an_ls': loss_an_ls,
    'wan': loss_wan,
    'epr': loss_epr,
    'role': loss_role,
    'EM': loss_EM,
    'EM_APL': loss_EM_APL,
    'EM_PL': loss_EM_PL,
    'EM_PL_ASL': loss_EM_PL_ASL
}


'''
top-level wrapper
'''


def compute_batch_loss(batch, P, Z):

    assert batch['preds'].dim() == 2

    batch_size = int(batch['preds'].size(0))
    num_classes = int(batch['preds'].size(1))

    loss_denom_mtx = (num_classes * batch_size) * torch.ones_like(batch['preds'])

    # input validation:
    assert torch.max(batch['label_vec_obs']) <= 1
    assert torch.min(batch['label_vec_obs']) >= -1
    assert batch['preds'].size() == batch['label_vec_obs'].size()
    assert P['loss'] in loss_functions

    # validate predictions:
    assert torch.max(batch['preds']) <= 1
    assert torch.min(batch['preds']) >= 0

    # compute loss for each image and class:
    loss_mtx, reg_loss = loss_functions[P['loss']](batch, P, Z)
    main_loss = (loss_mtx / loss_denom_mtx).sum()

    if reg_loss is not None:
        batch['loss_tensor'] = main_loss + reg_loss
        batch['reg_loss_np'] = reg_loss.clone().detach().cpu().numpy()
    else:
        batch['loss_tensor'] = main_loss
        batch['reg_loss_np'] = 0.0
    batch['loss_np'] = batch['loss_tensor'].clone().detach().cpu().numpy()

    return batch


'''
helper functions
'''


LOG_EPSILON = 1e-5


def neg_log(x):
    return - torch.log(x + LOG_EPSILON)

def log_loss(preds, targs):
    return targs * neg_log(preds)

def expected_positive_regularizer(preds, expected_num_pos, norm='2'):
    # Assumes predictions in [0,1].
    if norm == '1':
        reg = torch.abs(preds.sum(1).mean(0) - expected_num_pos)
    elif norm == '2':
        reg = (preds.sum(1).mean(0) - expected_num_pos)**2
    else:
        raise NotImplementedError
    return reg