import os
import copy
import time
import json
import numpy as np
import torch
import datasets
import models
import argparse
from tqdm import tqdm
from losses import compute_batch_loss
import datetime
from instrumentation import train_logger
import warnings
import torchvision.transforms as transforms
warnings.filterwarnings("ignore")


def run_train_phase(model, P, Z, logger, epoch, phase):
    
    '''
    Run one training phase.

    Parameters
    model: Model to train.
    P: Dictionary of parameters, which completely specify the training procedure.
    Z: Dictionary of temporary objects used during training.
    logger: Object used to track various metrics during training.
    epoch: Integer index of the current epoch.
    phase: String giving the phase name
    '''

    assert phase == 'train'
    model.train()

    desc = '[{}/{}]{}'.format(epoch, P['num_epochs'], phase.rjust(8, ' '))
    for batch in tqdm(Z['dataloaders'][phase], desc=desc, mininterval=1800):
        # move data to GPU:
        batch['image'] = batch['image'].to(Z['device'], non_blocking=True)
        
        desired_size = (336, 336)
        resize_transform = transforms.Resize(desired_size)
        resized_batch=[]
        resized_batch = torch.stack([resize_transform(img) for img in batch['image']])
        #print(resized_batch.shape)
        resized_batch.to(Z['device'], non_blocking=True)
        
        batch['labels_np'] = batch['label_vec_obs'].clone().numpy()  # copy of labels for use in metrics
        batch['label_vec_obs'] = batch['label_vec_obs'].to(Z['device'], non_blocking=True)
        # forward pass:
        Z['optimizer'].zero_grad()
        with torch.set_grad_enabled(True):
            batch['logits'], batch['logits_pl'], batch['similarity'] = model.f(batch['image'], resized_batch, P['txt_features'])
            batch['preds'] = torch.sigmoid(batch['logits'])
            if batch['preds'].dim() == 1:
                batch['preds'] = torch.unsqueeze(batch['preds'], 0)
             
            
            batch['preds_np'] = batch['preds'].clone().detach().cpu().numpy()  # copy of preds for use in metrics
            batch = compute_batch_loss(batch, P, Z)
        # backward pass:
        batch['loss_tensor'].backward()
        Z['optimizer'].step()
        # save current batch data:
        logger.update_phase_data(batch)


def run_eval_phase(model, P, Z, logger, epoch, phase):

    '''
    Run one evaluation phase.

    Parameters
    model: Model to train.
    P: Dictionary of parameters, which completely specify the training procedure.
    Z: Dictionary of temporary objects used during training.
    logger: Object used to track various metrics during training.
    epoch: Integer index of the current epoch.
    phase: String giving the phase name
    '''
    
    assert phase in ['val', 'test']
    model.eval()
    desc = '[{}/{}]{}'.format(epoch, P['num_epochs'], phase.rjust(8, ' '))
    for batch in tqdm(Z['dataloaders'][phase], desc=desc, mininterval=1800):
        # move data to GPU:
        batch['image'] = batch['image'].to(Z['device'], non_blocking=True)
        desired_size = (336, 336)
        resize_transform = transforms.Resize(desired_size)
        resized_batch=[]
        resized_batch = torch.stack([resize_transform(img) for img in batch['image']])
        resized_batch.to(Z['device'], non_blocking=True)
        
        batch['labels_np'] = batch['label_vec_obs'].clone().numpy()  # copy of labels for use in metrics
        batch['label_vec_obs'] = batch['label_vec_obs'].to(Z['device'], non_blocking=True)
        # forward pass:
        with torch.set_grad_enabled(False):
            
            batch['logits'], batch['logits_pl'], batch['similarity'] = model.f(batch['image'], resized_batch, P['txt_features'])
            batch['preds'] = torch.sigmoid(batch['logits'])
            if batch['preds'].dim() == 1:
                batch['preds'] = torch.unsqueeze(batch['preds'], 0)
            batch['preds_np'] = batch['preds'].clone().detach().cpu().numpy()  # copy of preds for use in metrics
            batch['loss_np'] = -1
            batch['reg_loss_np'] = -1
        # save current batch data:
        logger.update_phase_data(batch)


def train(model, P, Z):
    '''
    Train the model.

    Parameters
    P: Dictionary of parameters, which completely specify the training procedure.
    Z: Dictionary of temporary objects used during training.
    '''

    best_weights_f = copy.deepcopy(model.f.state_dict())
    logger = train_logger(P) # initialize logger
    if_early_stop = False

    for epoch_idx in range(0, P['num_epochs']):
        print('start epoch [{}/{}] ...'.format(epoch_idx + 1, P['num_epochs']))
        P['epoch'] = epoch_idx + 1
        for phase in ['train', 'val', 'test']:
            # reset phase metrics:
            logger.reset_phase_data()

            # run one phase:
            t_init = time.time()
            if phase == 'train':
                run_train_phase(model, P, Z, logger, P['epoch'], phase)
                #if P['epoch'] >= P['warmup_epoch'] and P['loss'] == 'EM_APL':
                    #aysmmetric_pseudo_labeling(model, P, Z, logger, P['epoch'], phase)
            else:
                run_eval_phase(model, P, Z, logger, P['epoch'], phase)

            # save end-of-phase metrics:
            logger.compute_phase_metrics(phase, P['epoch'])

            # print epoch status:
            logger.report(t_init, time.time(), phase, P['epoch'])

            # update best epoch, if applicable:
            new_best = logger.update_best_results(phase, P['epoch'], P['val_set_variant'])
            if new_best:
                print('*** new best weights ***')
                best_weights_f = copy.deepcopy(model.f.state_dict())
                #print('\nSaving best weights for f to {}/best_model_state.pt'.format(P['save_path']))
                #torch.save(best_weights_f, os.path.join(P['save_path'], '_best_model_state.pt'))
                
            '''
            elif (not new_best) and (phase == 'val'):
                print('*** early stop ***')
                if_early_stop = True
                break
            '''
        if if_early_stop:
            break

    print('')
    print('*** TRAINING COMPLETE ***')
    print('Best epoch: {}'.format(logger.best_epoch))
    print('Best epoch validation score: {:.2f}'.format(logger.get_stop_metric('val', logger.best_epoch, P['val_set_variant'])))
    print('Best epoch test score:       {:.2f}'.format(logger.get_stop_metric('test', logger.best_epoch, 'clean')))

    return P, model, logger, best_weights_f


def initialize_training_run(P, feature_extractor, linear_classifier):

    '''
    Set up for model training.
    Parameters
    P: Dictionary of parameters, which completely specify the training procedure.
    feature_extractor: Feature extractor model to start from.
    linear_classifier: Linear classifier model to start from.
    estimated_labels: NumPy array containing estimated training set labels to start from (for ROLE).
    '''
    
    np.random.seed(P['seed'])

    Z = {}

    # accelerator:
    #GPU=1
    #device = torch.device('cuda:'+str(GPU) if torch.cuda.is_available() else 'cpu')
    #text_features = np.load('VOC20text_feature.npy')
    
    Z['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    P['txt_features'] = torch.from_numpy(P['txt_features']).to(Z['device'])

    # data:
    Z['datasets'] = datasets.get_data(P)

    # observed label matrix:
    label_matrix = Z['datasets']['train'].label_matrix
    num_examples = int(np.shape(label_matrix)[0])
    mtx = np.array(label_matrix).astype(np.int8)
    total_pos = np.sum(mtx == 1)
    total_neg = np.sum(mtx == 0)
    print('training samples: {} total'.format(num_examples))
    print('true positives: {} total, {:.2f} per example on average.'.format(total_pos, total_pos / num_examples))
    print('true negatives: {} total, {:.2f} per example on average.'.format(total_neg, total_neg / num_examples))
    observed_label_matrix = Z['datasets']['train'].label_matrix_obs
    num_examples = int(np.shape(observed_label_matrix)[0])
    obs_mtx = np.array(observed_label_matrix).astype(np.int8)
    obs_total_pos = np.sum(obs_mtx == 1)
    obs_total_neg = np.sum(obs_mtx == -1)
    print('observed positives: {} total, {:.2f} per example on average.'.format(obs_total_pos, obs_total_pos / num_examples))
    print('observed negatives: {} total, {:.2f} per example on average.'.format(obs_total_neg, obs_total_neg / num_examples))

    # save dataset-specific parameters:
    P['num_classes'] = Z['datasets']['train'].num_classes
    
    
    # dataloaders:
    Z['dataloaders'] = {}
    for phase in ['train', 'val', 'test']:
        Z['dataloaders'][phase] = torch.utils.data.DataLoader(
            Z['datasets'][phase],
            batch_size = P['bsize'],
            shuffle = phase == 'train',
            sampler = None,
            num_workers = P['num_workers'],
            drop_last = False  # FIXME
        )

    # pseudo-labeling data:
    P['unlabel_num'] = []
    for i in range(observed_label_matrix.shape[1]):
        P['unlabel_num'].append(np.sum(observed_label_matrix[:, i] == 0))

    # model:
    model = models.MultilabelModel(P, Z, feature_extractor, linear_classifier)
    #model = models.MultilabelModel_baseline(P, Z, feature_extractor, linear_classifier)

    # optimization objects:
    f_params = [param for param in list(model.f.parameters()) if param.requires_grad]

    Z['optimizer'] = torch.optim.Adam(
        f_params,
        lr=P['lr']
    )

    return P, Z, model


def execute_training_run(P, feature_extractor, linear_classifier):

    '''
    Initialize, run the training process, and save the results.

    Parameters
    P: Dictionary of parameters, which completely specify the training procedure.
    feature_extractor: Feature extractor model to start from.
    linear_classifier: Linear classifier model to start from.
    estimated_labels: NumPy array containing estimated training set labels to start from (for ROLE).
    '''
    P, Z, model = initialize_training_run(P, feature_extractor, linear_classifier)
    model.to(Z['device'])

    P, model, logger, best_weights_f = train(model, P, Z)

    final_logs = logger.get_logs()
    model.f.load_state_dict(best_weights_f)

    return model.f.feature_extractor, model.f.linear_classifier, final_logs

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SPML_CLIP')
    parser.add_argument('-g', '--gpu', default='0', choices=['0', '1', '2', '3'], type=str)
    parser.add_argument('-d', '--dataset', default='pascal', choices=['pascal', 'coco', 'nuswide', 'cub'], type=str)
    parser.add_argument('-l', '--loss', default='EM_PL', choices=['bce', 'iun', 'an', 'EM', 'EM_APL', 'EM_PL', 'EM_PL_ASL'], type=str)
    parser.add_argument('-m', '--model', default='resnet50', choices=['clip_vision','resnet50', 'convnext_xlarge_22k', 'convnext_xlarge_1k'], type=str)
    parser.add_argument('-t', '--temp', default=0.01, type=float)
    parser.add_argument('-th', '--threshold', default=0.3, type=float)
    parser.add_argument('-p', '--partial', default=0.0, type=float)
    parser.add_argument('-s', '--pytorch_seed', default=0, type=int)  # try 0, 1, 8
    
    args = parser.parse_args()

    P = {}

    # Top-level parameters:
    P['GPU'] = args.gpu
    P['dataset'] = args.dataset
    P['loss'] = args.loss
    P['val_set_variant'] = 'clean'  # clean, observed
    P['test_set_variant'] = 'clean' # clean, observed
    # System parameters:
    os.environ["CUDA_VISIBLE_DEVICES"] = P['GPU']
    P['pytorch_seed'] = args.pytorch_seed
    torch.manual_seed(P['pytorch_seed'])
    torch.cuda.manual_seed(P['pytorch_seed'])
    
    # Optimization parameters:
    if P['dataset'] == 'pascal':
        P['bsize'] = 8 #8 for resnet50, 6 for ViT-L
        P['lr'] = 1e-5 
        P['warmup_epoch'] = 0
        P['alpha'] = 0.2
        P['beta_pos'] = 0.7  #0.7
        P['beta_neg'] = 0.0 
        P['unknown']  = 4.0  #P['alpha'] = 0.2
        P['positive'] = 2.0  #P['beta_neg'] = 0.0 
        P['negative'] = 4.0  #P['beta_pos'] = 0.2 
        P['txt_features'] = np.load('VOC20text_feature_labelonly.npy')
        P['partial'] = args.partial #[0.1, 0.2, 0.3, 0.4]
        
        P['temp'] = args.temp #[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
        P['threshold'] = args.threshold #[0.1, 0.15, 0.2, 0.25, 0.3]
        
    elif P['dataset'] == 'cub':
        P['bsize'] = 8 #8 for resnet50, 6 for ViT-L
        P['lr'] = 1e-4 #1e-4 resnet50
        P['warmup_epoch'] = 0
        P['unknown']  = 4.0  #P['alpha'] = 0.2
        P['positive'] = 2.0  #P['beta_neg'] = 0.0 
        P['negative'] = 4.0
        P['alpha'] = 0.01
        P['beta_pos'] = 0.7
        P['beta_neg'] = 0.0 #0.2
        P['txt_features'] = np.load('CUB312text_feature.npy') 
        P['partial'] = 0.0
        
        P['temp'] = args.temp #[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
        P['threshold'] = args.threshold #[0.1, 0.15, 0.2, 0.25, 0.3]
        
    elif P['dataset'] == 'nuswide':
        P['bsize'] = 8 #8 for resnet50, 6 for Vit-L
        P['lr'] = 1e-5
        P['warmup_epoch'] = 0
        P['unknown']  = 4  #P['alpha'] = 0.2
        P['positive'] = 2  #P['beta_neg'] = 0.0 
        P['negative'] = 4
        P['alpha'] = 0.1
        P['beta_pos'] = 0.7
        P['beta_neg'] = 0.0 
        P['partial'] = 0.0
        P['txt_features'] = np.load('NUS81text_feature.npy') 
        
        P['temp'] = args.temp #[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
        P['threshold'] = args.threshold #[0.1, 0.15, 0.2, 0.25, 0.3]
        
    elif P['dataset'] == 'coco':
        P['bsize'] = 8 #8 for resnet50, 6 for ViT-L
        P['lr'] = 1e-5 
        P['warmup_epoch'] = 0
        P['unknown']  = 4  #P['alpha'] = 0.2
        P['positive'] = 2  #P['beta_neg'] = 0.0 
        P['negative'] = 4  #P['beta_pos'] = 0.2 
        P['alpha'] = 0.1
        P['beta_pos'] = 0.7
        P['beta_neg'] = 0.0 
        P['partial'] = 0.0
        P['txt_features'] = np.load('CoCo80text_feature_labelonly.npy')
        
        P['temp'] = args.temp #[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
        P['threshold'] = args.threshold ##[0.1, 0.15, 0.2, 0.25, 0.3]
        
       

    # Additional parameters:
    P['seed'] = 1200  # overall numpy seed
    P['use_pretrained'] = True  # True, False
    P['num_workers'] = 8
    P['stop_metric'] = 'map'  # metric used to select the best epoch

    # Dataset parameters:
    P['split_seed'] = 1200  # seed for train/val splitting
    P['val_frac'] = 0.2  # fraction of train set to split off for val
    P['ss_seed'] = 999  # seed for subsampling
    P['ss_frac_train'] = 1.0  # fraction of training set to subsample
    P['ss_frac_val'] = 1.0  # fraction of val set to subsample

    # Dependent parameters:
    if P['loss'] == 'bce':
        P['train_set_variant'] = 'clean'
    else:
        P['train_set_variant'] = 'observed'

    # training parameters:
    P['num_epochs'] = 10
    P['freeze_feature_extractor'] = False
    P['use_feats'] = False
    P['arch'] = args.model #{'clip_vision','resnet50', 'convnext_xlarge_22k', 'convnext_xlarge_1k','clip_vision_1k+12k'}
    #P['feature_extractor_arch'] = 'resnet50'
    #P['feat_dim'] = 2048 
    P['save_path'] = './results/' + P['dataset'] + P['arch']
    # run training process:
    print('[{} + {}] start exp ...'.format(P['dataset'], P['loss']))
    print("P is: ", P)
   
    (feature_extractor, linear_classifier, logs) = execute_training_run(P, feature_extractor=None, linear_classifier=None)
