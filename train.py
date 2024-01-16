import os
import sys
import numpy as np
import time
import torch
import utils
from tqdm import tqdm
import glob
import logging
import argparse
import torch.nn as nn
import genotypes as genotypes
import torch.utils
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from thop import profile
# from old_model_py.model import NetworkCIFAR as Network
from model import NetworkCIFAR as Network
from dataset import CCBMDataset
from modules import data_setup
from model_parameters import PH2_DLV3_FT_params, Derm7pt_Manually_params
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, hamming_loss, precision_score, recall_score, f1_score, roc_curve, auc
from torchvision.models import resnet101
import torchvision
from modules.model_builder import CCBM
import model_params as params1
import torch.nn.functional as F



parser = argparse.ArgumentParser("training cifar-10")
parser.add_argument('--workers', type=int, default=16, help='number of workers to load dataset')
parser.add_argument('--data', type=str, default='datapath', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='choose dataset')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='exp', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DrNAS_cifar10', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

args.dataset = 'ph2'
args.dataset = 'derm7'  

data_paths = {
    "cifar10": "./",
    "cifar100": "./",
    "ph2": "./",
    "derm7": "./",
    #"ImageNet16-120": "/ssd1/ImageNet16",
    #"imagenet-1k": "/ssd2/chenwy/imagenet_final",
}


data_path=data_paths[args.dataset]

args.data = data_path

args.concept_loss_weight = 0.6
args.auxiliary_weight = 0

# set up logs
args.save = './experiments/{}/eval-{}-{}-{}-{}'.format(
    args.dataset, args.save, time.strftime("%Y%m%d-%H%M%S"), args.arch, args.seed)
if args.cutout:
    args.save += '-cutout-' + str(args.cutout_length)
if args.auxiliary:
    args.save += '-auxiliary-' + str(args.auxiliary_weight)
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
writer = SummaryWriter(args.save)

CIFAR_CLASSES = 10

if args.dataset=='cifar100':
    CIFAR_CLASSES = 100
elif args.dataset=='derm7':
    CIFAR_CLASSES = 2

def main():
    

    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    logging.info(args.data)
    genotype = eval("genotypes.%s" % args.arch)
    print('---------Genotype---------')
    logging.info(genotype)
    print('--------------------------')
    print(f"args.layers:    ", args.layers)
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
    model.drop_path_prob = 0
    macs, params = profile(model, inputs=(torch.randn(1, 3, 32, 32), ), verbose=False)
    logging.info("param = %f, flops = %f", params, macs)
    #-----------------------
    #-------------------------
    #----------------------------

    class_ind_vec = np.load(params1.CLASS_INDICATOR_VECTORS).T
    dtype = torch.FloatTensor
    resnet101_weights = torchvision.models.ResNet101_Weights.IMAGENET1K_V1
    resnet101 = torchvision.models.resnet101(weights=resnet101_weights)
    resnet101 = nn.Sequential(
        *list(resnet101.children())[:-3]
    )

    # Create the CCbm
    ccbm = CCBM(input_shape=1024,
                num_concepts=params1.NUM_CONCEPTS,
                num_classes=params1.NUM_CLASSES)

    # Initialize FC layer with custom weights
    init = np.multiply(0.1 * np.ones((params1.NUM_CONCEPTS, params1.NUM_CLASSES)), class_ind_vec)
    ccbm.classifier[0].weight.data = torch.from_numpy(init).type(dtype).T

    # Add the CCbm to the backbone model
    model = nn.Sequential(resnet101, ccbm).cuda()
    #-----------------------
    #-------------------------
    #----------------------------

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion_unique = nn.BCEWithLogitsLoss()
    criterion = criterion.cuda()
    criterion_unique = criterion_unique.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
        )
    
    train_transform = A.Compose([
        A.PadIfNeeded(512, 512),
        A.CenterCrop(width=512, height=512),
        A.Resize(width=32, height=32),  # (299, 299) for inception; (224,224) for others
        A.RandomRotate90(),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    valid_transform = A.Compose([
        A.PadIfNeeded(512, 512),
        A.CenterCrop(width=512, height=512),
        A.Resize(width=32, height=32),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    test_transform = A.Compose([
        A.PadIfNeeded(512, 512),
        A.CenterCrop(width=512, height=512),
        A.Resize(width=32, height=32),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    
    #train_transform, valid_transform = utils._data_transforms_cifar10(args)
    if args.dataset == 'cifar100':
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
    elif args.dataset == 'ph2':
        train_data, valid_data, train_queue, valid_queue = data_setup.create_dataloaders(params=PH2_DLV3_FT_params,
                                                            train_transform=train_transform,
                                                            val_transform=valid_transform)
        test_queue, _ = data_setup.create_dataloader_for_evaluation(params=PH2_DLV3_FT_params,
                                                                             transform=test_transform)
    elif args.dataset == 'derm7':
        train_data, valid_data, train_queue, valid_queue = data_setup.create_dataloaders(params=Derm7pt_Manually_params,
                                                            train_transform=train_transform,
                                                            val_transform=valid_transform)
        test_queue, _ = data_setup.create_dataloader_for_evaluation(params=Derm7pt_Manually_params,
                                                                             transform=test_transform)
    else:
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    #train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    #valid_queue = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    best_acc = train_acc = valid_acc = 0.0
    epoch_bar = tqdm(range(args.epochs), position=0, leave=True)
    for epoch in epoch_bar:
        # logging.info('epoch %d lr %e', epoch, scheduler.get_last_lr()[0])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        description = 'Epoch [{}/{}] | Train:{} | Validation:{} | Best: {}'.format(epoch+1, args.epochs, train_acc, valid_acc, best_acc)

        train_acc, train_obj, uniqueness_loss = train(train_queue, model, criterion, criterion_unique, optimizer)
        # logging.info('train_acc %f', train_acc)
        description = 'Epoch [{}/{}] | Train:{} | Validation:{} | Best: {} | U_loss: {} | loss: {}'.format(epoch+1, args.epochs, train_acc, valid_acc, best_acc, uniqueness_loss, train_obj)
        epoch_bar.set_description(description)

        valid_acc, valid_obj = infer(valid_queue, model, criterion, criterion_unique)
        if valid_acc > best_acc:
            best_acc = valid_acc
        # logging.info('valid_acc %f, best_acc %f', valid_acc, best_acc)
        description = 'Epoch [{}/{}] | Train:{} | Validation:{} | Best: {} | U_loss: {} | loss: {}'.format(epoch+1, args.epochs, train_acc, valid_acc, best_acc, uniqueness_loss, train_obj)
        epoch_bar.set_description(description)

        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("acc/valid_best", best_acc, epoch)
        writer.add_scalar("acc/valid", valid_acc, epoch)

        scheduler.step()
        utils.save(model, os.path.join(args.save, 'weights.pt'))
    
    test_acc, test_obj, test_f1s = test(test_queue, model, criterion, criterion_unique)
    #test_acc, test_obj = test(test_queue, model, criterion, criterion_unique)
    #train_acc, train_obj, train_f1s = test(train_queue, model, criterion)
    #val_acc, val_obj, val_f1s = test(valid_queue, model, criterion)

    #print(f"test accuracy:  {test_acc}\ntest objective: {test_obj}\n")
    print(f"test accuracy:  {test_acc}\ntest objective: {test_obj}\nf1 score on test concepts:  {test_f1s}")


def train(train_queue, model, criterion, criterion_unique, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    objs_uniqueness = utils.AvgrageMeter()
    model.train()

    for step, (input, target, indicator_vector) in enumerate(train_queue):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        indicator_vector = indicator_vector.type(torch.FloatTensor)

        indicator_vector = indicator_vector.cuda(non_blocking=True)
        optimizer.zero_grad()
        logits, logits_aux, logits8 = model(input)
        encoded_targets = F.one_hot(target,2).float()
        #print(target)
        #print(encoded_targets, logits)
        #logits = model(input)
        #logits, logits_aux = model(input)
        #print(logits)
        #print(indicator_vector)
        loss = criterion(logits, encoded_targets)
        uniqueness_loss =  criterion_unique(logits_aux, indicator_vector)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            #loss += args.auxiliary_weight * loss_aux
            loss += args.concept_loss_weight * uniqueness_loss

        #print(f"auxiliary_weight", args.auxiliary_weight)
        #print(f"concept_loss_weight", args.concept_loss_weight)
        
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 2))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)
        objs_uniqueness.update(uniqueness_loss.data, n)
    #print(f"loss:   ",loss)
    #print(f"args.concept_loss_weight * uniqueness_loss:    ", args.concept_loss_weight * uniqueness_loss)
    #print(f"args.auxiliary_weight * loss_aux:   ", args.auxiliary_weight * loss_aux)
    return top1.avg, objs.avg, objs_uniqueness.avg


def infer(valid_queue, model, criterion, criterion_unique):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target, indicator_vector) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)
            indicator_vector = indicator_vector.type(torch.FloatTensor)

            indicator_vector = indicator_vector.cuda(non_blocking=True)

            logits, logits_aux, logits8 = model(input)
            #logits = model(input)
            encoded_targets = F.one_hot(target,2).float()
            loss = criterion(logits, encoded_targets)

            #logits, logits_aux = model(input)
            #loss = criterion(logits, target)
            #loss = criterion_unique(logits, indicator_vector)


            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 2))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

    return top1.avg, objs.avg

def test(test_queue, model, criterion, criterion_unique):
    print(test_queue)
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    predicted = []
    targets = []

    with torch.no_grad():
        for step, (input, target, indicator_vector) in enumerate(test_queue):
            #print(f"input:  ",input)
            #print(f"target: ",target)
            #print(f"indicator_vector:   ",indicator_vector)
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            indicator_vector = indicator_vector.type(torch.FloatTensor)

            indicator_vector = indicator_vector.cuda(non_blocking=True)

            logits, logits_aux, logits8 = model(input)
            #logits = model(input)

            #logits, logits_aux = model(input)
            loss = criterion(logits, target)
            #loss = criterion_unique(logits_aux, indicator_vector)

            targets.append(indicator_vector.squeeze().cpu().numpy())
            predicted.append(torch.where(torch.tanh(logits_aux) > 0.7, 1, 0).squeeze().cpu().numpy())

            # print(f"logits8:  ",logits8)
            #

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 2))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

    
    print("-----------------end---------------\n-----------------next---------------")
    #print(f"targets[0].shape:    ", targets[0].shape)
    #print(f"predicted[0].shape:    ", predicted[0].shape)
    #print(f"len(predicted[0].shape):    ", len(predicted[0].shape))
    print(f"-------------------------------------\ntargets\n--------------------------------\n",targets[:30])
    print(f"-------------------------------------\npredicted\n--------------------------------\n",predicted[:30])
    #print(len(targets))
    #print(len(predicted))

    #print(f"targets.shape:    ", targets.shape)
    #print(f"predicted.shape:    ", predicted.shape)  
    
    f1s = np.array([])
    f1s_final = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    if len(predicted[0].shape) == 1:
         print('in!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #print("f1_score(targets, predicted, average=None):  ",f1_score(targets, predicted, average=None))
         f1s_final += np.array(f1_score(targets, predicted, average=None))

    
    ''' else:
        f1s_final = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        for j in range(len(targets)):
            if len(targets[j]) != 8:
                #print(f"targets[j]:    ", targets[j])
                #print(f"predicted[j]:    ", predicted[j])
                #print("f1_score(targets[j], predicted[j], average=None):  ",f1_score(targets[j], predicted[j], average=None))
                f1s_final += np.array(f1_score(targets[j], predicted[j], average=None))
                #print(j)
    
        f1s_final = f1s_final / len(targets)
    '''
    print(f"f1_score_final:   ", f1s_final)
    #print("len(f1s_final):  ",len(f1s_final))
                 

    
            
    return top1.avg, objs.avg, f1s_final

if __name__ == '__main__':
    main()
