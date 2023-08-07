import torch
import numpy as np
import copy
from torchvision.models import resnet50
from collections import OrderedDict
from transformers import CLIPProcessor, CLIPVisionModel, CLIPVisionModelWithProjection, AutoTokenizer, CLIPTextModelWithProjection

from torch import nn
import torch.nn.functional as F
import timm
import warnings
warnings.filterwarnings("ignore")



'''
model definitions
'''
class FCNet(torch.nn.Module):
    def __init__(self, num_feats, num_classes):
        super(FCNet, self).__init__()
        self.fc = torch.nn.Linear(num_feats, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x
    
    
class Adaptor(torch.nn.Module):
    def __init__(self, num_feats=768, num_hidden=384):
        super(Adaptor, self).__init__()
        self.fc1 = torch.nn.Linear(num_feats, num_hidden)
        self.fc2 = torch.nn.Linear(num_hidden, num_feats)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class ImageClassifier(torch.nn.Module):

    def __init__(self, P, Z, model_feature_extractor=None, model_linear_classifier=None):

        super(ImageClassifier, self).__init__()
        print('initializing image classifier')

        model_feature_extractor_in = copy.deepcopy(model_feature_extractor)
        model_linear_classifier_in = copy.deepcopy(model_linear_classifier)

        self.arch = P['arch']
            
        if self.arch == 'clip_vision':
            print('training CLIP_Vision Encoder')
            self.feature_extractor = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14-336",output_attentions=True)
            for p in self.feature_extractor.parameters():
                p.requires_grad = False
            #CLIP-ViT-L
            self.vision_extractor = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14-336",output_attentions=True)
            for p in self.vision_extractor.parameters():
                p.requires_grad = True
            '''   
            self.text_extractor =  CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14-336",output_attentions=True)
            for p in self.text_extractor.parameters():
                p.requires_grad = True
            self.tokenizer = AutoTokenizer.from_pretrained
            
            '''   
            #self.projection = nn.Linear(in_features=1024, out_features=768, bias=True)            
            #self.img_to_latents = EmbedToLatents(768, 384)
            self.adaptor = Adaptor(768, 384)
            self.linear_classifier = FCNet(768, P['num_classes'])
            #self.threshold_up = torch.nn.Parameter(torch.randn(1)) 
            self.threshold = P['threshold']
            self.scalar_pos = torch.tensor(1, dtype=torch.float32).to(Z['device'])
            self.scalar_neg = torch.tensor(-1, dtype=torch.float32).to(Z['device'])
            #self.adaptor = Adaptor(768, 384)
            self.temperature = P['temp']
            self.partial = P['partial']
            self.classes = P['num_classes']
            self.device = Z['device']
        elif self.arch == 'resnet50':
            feature_extractor = resnet50(pretrained=True)
            feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1])
            feature_extractor.avgpool = torch.nn.AdaptiveAvgPool2d(1)
            self.vision_extractor = feature_extractor
            for param in self.vision_extractor.parameters():
                    param.requires_grad = True
            
            self.feature_extractor = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14-336",output_attentions=True)
            for p in self.feature_extractor.parameters():
                p.requires_grad = False
            self.linear_classifier = FCNet(2048, P['num_classes'])
            
            self.threshold = P['threshold']
            self.scalar_pos = torch.tensor(1, dtype=torch.float32).to(Z['device'])
            self.scalar_neg = torch.tensor(-1, dtype=torch.float32).to(Z['device'])
            #self.adaptor = Adaptor(768, 384)
            self.temperature = P['temp']
            self.partial = P['partial']
            self.classes = P['num_classes']
            self.device = Z['device']
            
        elif self.arch == 'convnext_xlarge_22k':
            self.vision_extractor = timm.create_model('convnext_xlarge.fb_in22k', pretrained=True)
            self.vision_extractor.head.fc=nn.Linear(in_features=2048, out_features=P['num_classes'], bias=True)
            for param in self.vision_extractor.parameters():
                    param.requires_grad = True
            
            self.feature_extractor = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14-336",output_attentions=True)
            for p in self.feature_extractor.parameters():
                p.requires_grad = False
            #self.linear_classifier = FCNet(2048, P['num_classes'])
            
            self.threshold = P['threshold']
            self.scalar_pos = torch.tensor(1, dtype=torch.float32).to(Z['device'])
            self.scalar_neg = torch.tensor(-1, dtype=torch.float32).to(Z['device'])
            #self.adaptor = Adaptor(768, 384)
            self.temperature = P['temp']
            self.partial = P['partial']
            self.classes = P['num_classes']
            self.device = Z['device']
            
        elif self.arch == 'convnext_xlarge_1k':
            self.vision_extractor = timm.create_model('convnext_xlarge.fb_in22k_ft_in1k', pretrained=True)
            self.vision_extractor.head.fc=nn.Linear(in_features=2048, out_features=P['num_classes'], bias=True)
            for param in self.vision_extractor.parameters():
                param.requires_grad = True
            
            self.feature_extractor = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14-336",output_attentions=True)
            for p in self.feature_extractor.parameters():
                p.requires_grad = False
            #self.linear_classifier = FCNet(2048, P['num_classes'])
            
            self.threshold = P['threshold']
            self.scalar_pos = torch.tensor(1, dtype=torch.float32).to(Z['device'])
            self.scalar_neg = torch.tensor(-1, dtype=torch.float32).to(Z['device'])
            #self.adaptor = Adaptor(768, 384)
            self.temperature = P['temp']
            self.partial = P['partial']
            self.classes = P['num_classes']
            self.device = Z['device']
            
        else:
            raise ValueError('Architecture not implemented.')
            
    def temperature_scaled_softmax(self, logits, temperature):
        return F.softmax(logits / temperature, dim=-1)
    
    def forward(self, x, xs, y):
        if self.arch == 'clip_vision':
            # x is a batch of images
            feats = self.feature_extractor(xs)
            image_embedding = feats.image_embeds 
            image_embedding = F.normalize(image_embedding, dim=-1)
            similarity = image_embedding @ y.T
            
            #The similarity should be a T-softmax
            similarity = self.temperature_scaled_softmax(similarity, temperature=self.temperature)
            #print("similarity: ", similarity)
            
            
            #options 1, setting a threshold to determin final pseduo-label (-1, 1) (too aggressive, not work)
            #logits_pl = torch.where(similarity > self.threshold_up, self.scalar_pos, self.scalar_neg)
            
            '''
            #options 2, setting a thresholds to determin only final positive pseduo-label (0, 1)
            logits_pl = torch.zeros_like(similarity)
            logits_pl = torch.where(similarity > self.threshold_up, self.scalar_pos, logits_pl)
            #logits_pl = torch.where(similarity < self.threshold_down, self.scalar_neg, logits_pl)
            #print("logits_pl: ", logits_pl)
            '''
            #options 3, setting two thresholds to determin the final pos./neg. pseduo-label (-1, 0, 1)
            logits_pl = torch.zeros_like(similarity)
            #positive pseudo label
            logits_pl = torch.where(similarity > self.threshold, self.scalar_pos, logits_pl)
            #negative pseudo label
            num_elements = int(self.partial * self.classes)
            #print("num_elements: ", num_elements)
            _, smallest_indices = torch.topk(similarity, num_elements, largest=False)
            #print("smallest_indices: ", smallest_indices)
            values = torch.full(smallest_indices.shape, -1, dtype=logits_pl.dtype).to(self.device)
            logits_pl.scatter_(1, smallest_indices, values)
            #print("logits_pl: ", logits_pl)
            
            vision = self.vision_extractor(xs)
            #vision = self.adaptor(vision.image_embeds)
            #logits = self.linear_classifier(vision)
            logits = self.linear_classifier(vision.image_embeds)
            
            #print("logits: ", logits)
        elif self.arch == 'resnet50':
            # x is a batch of images
            feats = self.feature_extractor(xs)
            image_embedding = feats.image_embeds 
            image_embedding = F.normalize(image_embedding, dim=-1)
            similarity = image_embedding @ y.T
            #print("similarity: ", similarity.shape)
            #The similarity should be a T-softmax
            similarity = self.temperature_scaled_softmax(similarity, temperature=self.temperature)
            #print("similarity: ", similarity)
            
            #options 1, setting a threshold to determin final pseduo-label (-1, 1) (too aggressive, not work)
            #logits_pl = torch.where(similarity > self.threshold_up, self.scalar_pos, self.scalar_neg)
            
            '''
            #options 2, setting a thresholds to determin only final positive pseduo-label (0, 1)
            logits_pl = torch.zeros_like(similarity)
            logits_pl = torch.where(similarity > self.threshold_up, self.scalar_pos, logits_pl)
            #logits_pl = torch.where(similarity < self.threshold_down, self.scalar_neg, logits_pl)
            #print("logits_pl: ", logits_pl)
            '''
            
            #options 3, setting two thresholds to determin the final pos./neg. pseduo-label (-1, 0, 1)
            logits_pl = torch.zeros_like(similarity)
            #positive pseudo label
            logits_pl = torch.where(similarity > self.threshold, self.scalar_pos, logits_pl)
            
            '''
            #negative pseudo label
            num_elements = int(self.partial * self.classes)
            #print("num_elements: ", num_elements)
            _, smallest_indices = torch.topk(similarity, num_elements, largest=False)
            #print("smallest_indices: ", smallest_indices)
            values = torch.full(smallest_indices.shape, -1, dtype=logits_pl.dtype).to(self.device)
            logits_pl.scatter_(1, smallest_indices, values)
            '''
            vision = self.vision_extractor(x)
            #print("logits_pl: ", logits_pl)
            logits = self.linear_classifier(torch.squeeze(vision))
        elif self.arch == 'convnext_xlarge_22k':
            # x is a batch of images
            feats = self.feature_extractor(xs)
            image_embedding = feats.image_embeds 
            image_embedding = F.normalize(image_embedding, dim=-1)
            similarity = image_embedding @ y.T
            #print("similarity: ", similarity.shape)
            #The similarity should be a T-softmax
            similarity = self.temperature_scaled_softmax(similarity, temperature=self.temperature)
            #print("similarity: ", similarity)
            
            #options 1, setting a threshold to determin final pseduo-label (-1, 1) (too aggressive, not work)
            #logits_pl = torch.where(similarity > self.threshold_up, self.scalar_pos, self.scalar_neg)
            
            '''
            #options 2, setting a thresholds to determin only final positive pseduo-label (0, 1)
            logits_pl = torch.zeros_like(similarity)
            logits_pl = torch.where(similarity > self.threshold_up, self.scalar_pos, logits_pl)
            #logits_pl = torch.where(similarity < self.threshold_down, self.scalar_neg, logits_pl)
            #print("logits_pl: ", logits_pl)
            '''
            
            #options 3, setting two thresholds to determin the final pos./neg. pseduo-label (-1, 0, 1)
            logits_pl = torch.zeros_like(similarity)
            #positive pseudo label
            logits_pl = torch.where(similarity > self.threshold, self.scalar_pos, logits_pl)
            
            '''
            #negative pseudo label
            num_elements = int(self.partial * self.classes)
            #print("num_elements: ", num_elements)
            _, smallest_indices = torch.topk(similarity, num_elements, largest=False)
            #print("smallest_indices: ", smallest_indices)
            values = torch.full(smallest_indices.shape, -1, dtype=logits_pl.dtype).to(self.device)
            logits_pl.scatter_(1, smallest_indices, values)
            '''
            logits = self.vision_extractor(x)
        elif self.arch == 'convnext_xlarge_1k':
            # x is a batch of images
            feats = self.feature_extractor(xs)
            image_embedding = feats.image_embeds 
            image_embedding = F.normalize(image_embedding, dim=-1)
            similarity = image_embedding @ y.T
            #print("similarity: ", similarity.shape)
            #The similarity should be a T-softmax
            similarity = self.temperature_scaled_softmax(similarity, temperature=self.temperature)
            #print("similarity: ", similarity)
            
            #options 1, setting a threshold to determin final pseduo-label (-1, 1) (too aggressive, not work)
            #logits_pl = torch.where(similarity > self.threshold_up, self.scalar_pos, self.scalar_neg)
            
            '''
            #options 2, setting a thresholds to determin only final positive pseduo-label (0, 1)
            logits_pl = torch.zeros_like(similarity)
            logits_pl = torch.where(similarity > self.threshold_up, self.scalar_pos, logits_pl)
            #logits_pl = torch.where(similarity < self.threshold_down, self.scalar_neg, logits_pl)
            #print("logits_pl: ", logits_pl)
            '''
            
            #options 3, setting two thresholds to determin the final pos./neg. pseduo-label (-1, 0, 1)
            logits_pl = torch.zeros_like(similarity)
            #positive pseudo label
            logits_pl = torch.where(similarity > self.threshold, self.scalar_pos, logits_pl)
            
            '''
            #negative pseudo label
            num_elements = int(self.partial * self.classes)
            #print("num_elements: ", num_elements)
            _, smallest_indices = torch.topk(similarity, num_elements, largest=False)
            #print("smallest_indices: ", smallest_indices)
            values = torch.full(smallest_indices.shape, -1, dtype=logits_pl.dtype).to(self.device)
            logits_pl.scatter_(1, smallest_indices, values)
            '''
            logits = self.vision_extractor(x)
        else:
            # x is a batch of images
            feats = self.resnet50(x)
            logits = self.linear_classifier(torch.squeeze(feats))
            
        return logits, logits_pl, similarity
    

class MultilabelModel(torch.nn.Module):
    def __init__(self, P, Z, feature_extractor, linear_classifier):
        super(MultilabelModel, self).__init__()
        self.f = ImageClassifier(P, Z, feature_extractor, linear_classifier)

    def forward(self, batch):
        f_logits = self.f(batch['image'])
        return f_logits
    
#Baseline Model
class ImageClassifier_baseline(torch.nn.Module):

    def __init__(self, P, model_feature_extractor=None, model_linear_classifier=None):

        super(ImageClassifier_baseline, self).__init__()
        print('initializing image classifier')

        model_feature_extractor_in = copy.deepcopy(model_feature_extractor)
        model_linear_classifier_in = copy.deepcopy(model_linear_classifier)
        self.arch = P['arch']
        if self.arch == 'resnet50':
            # configure feature extractor:
            if model_feature_extractor_in is not None:
                print('feature extractor: specified by user')
                feature_extractor = model_feature_extractor_in
            else:
                if P['use_pretrained']:
                    print('feature extractor: imagenet pretrained')
                    feature_extractor = resnet50(pretrained=True)
                else:
                    print('feature extractor: randomly initialized')
                    feature_extractor = resnet50(pretrained=False)
                feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1])
            if P['freeze_feature_extractor']:
                print('feature extractor frozen')
                for param in feature_extractor.parameters():
                    param.requires_grad = False
            else:
                print('feature extractor trainable')
                for param in feature_extractor.parameters():
                    param.requires_grad = True
            feature_extractor.avgpool = torch.nn.AdaptiveAvgPool2d(1)
            self.feature_extractor = feature_extractor

            # configure final fully connected layer:
            if model_linear_classifier_in is not None:
                print('linear classifier layer: specified by user')
                linear_classifier = model_linear_classifier_in
            else:
                print('linear classifier layer: randomly initialized')
                linear_classifier = torch.nn.Linear(P['feat_dim'], P['num_classes'], bias=True)
            self.linear_classifier = linear_classifier

        elif self.arch == 'linear':
            print('training a linear classifier only')
            self.feature_extractor = None
            self.linear_classifier = FCNet(P['feat_dim'], P['num_classes'])
        elif self.arch == 'clip_vision_baseline':
            print('training CLIP_Vision Encoder') #
            self.feature_extractor = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14-336",output_attentions=True)
            for p in self.feature_extractor.parameters():
                p.requires_grad = True
            self.linear_classifier = FCNet(768, P['num_classes'])
            self.threshold_up = P['threshold']
            self.scalar_pos = torch.tensor(1, dtype=torch.float32).to(Z['device'])
            self.temperature = 0.1
            self.classes = P['num_classes']
            self.device = Z['device']
            
        else:
            raise ValueError('Architecture not implemented.')

    def forward(self, x):
        
        if self.arch == 'linear':
            # x is a batch of feature vectors
            logits = self.linear_classifier(x)
        elif self.arch == 'clip_vision_baseline':
            feats = self.feature_extractor(x)
            logits = self.linear_classifier(feats.image_embeds)#feats.pooler_output
        else:
            # x is a batch of images
            feats = self.feature_extractor(x)
            logits = self.linear_classifier(torch.squeeze(feats))
            
        return logits
    
class MultilabelModel_baseline(torch.nn.Module):
    def __init__(self, P, feature_extractor, linear_classifier):
        super(MultilabelModel_baseline, self).__init__()
        self.f = ImageClassifier_baseline(P, feature_extractor, linear_classifier)

    def forward(self, batch):
        f_logits = self.f(batch['image'])
        return f_logits
