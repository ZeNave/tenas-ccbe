# What is the accuracy of a classifier when it receives the exact concepts in the image

from modules import data_setup
from model_parameters import Derm7pt_Manually_params
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.utils
import utils
from torch import functional as F

class Classifier(nn.Module):
    def __init__(self, in_features, num_features,  num_classes) -> None:
        super(Classifier, self).__init__()
        self.in_features = in_features
        self.num_feature = num_features
        self.num_classes = num_classes
        self.layers = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=num_classes),
            #nn.ReLU(),
            #nn.Linear(in_features=num_features, out_features=num_features),
            #nn.ReLU(),
            #nn.Linear(in_features=num_features, out_features=num_classes),
            #nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

device = 'cuda'

model = Classifier(8, 64, 2).to(device)
loss = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    0.025, #learning_rate=args.learning_rate
    0.9, # momentum=args.momentum
    3e-4, # weight_decay=args.weight_decay
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

train_data, valid_data, train_queue, valid_queue = data_setup.create_dataloaders(params=Derm7pt_Manually_params,
                                                            train_transform=train_transform,
                                                            val_transform=valid_transform)
test_queue, _ = data_setup.create_dataloader_for_evaluation(params=Derm7pt_Manually_params,
                                                                        transform=test_transform)
top1 = utils.AvgrageMeter()
epochs = 100

for epoch in range(epochs):
    print(f"-----------------EPOCHS-----------", epoch)
    step = 0
    loss_sum = 0

    for step, (input, target, indicator_vector) in enumerate(train_queue):
        step+=1
        #input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        indicator_vector = indicator_vector.type(torch.FloatTensor)

        indicator_vector = indicator_vector.cuda(non_blocking=True)
        optimizer.zero_grad()
        #logits, logits_aux, logits8 = model(input)
        #logits, logits_aux = model(input)
        #loss = criterion(logits, target)
        #uniqueness_loss =  criterion_unique(logits8, indicator_vector)
        predictions = model(indicator_vector)

        #predictions = torch.argmax(predictions, 1, keepdim=False)

        predictions = predictions.float()
        predictions.requires_grad_(True)

        loss_output = loss(predictions, target)
        
        loss_output.backward()

        #print(f"loss_output:   ",loss_output)
        
        # nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec2 = utils.accuracy(predictions, target, topk=(1, 2))
        # n = input.size(0)
        # objs.update(loss.data, n)
        # top1.update(prec1.data, n)
        # top5.update(prec5.data, n)
        loss_sum += loss_output
        n = input.size(0)
        top1.update(prec1.data, n)
    loss_step = loss_sum/step
    print(f"loss_output:    ", loss_step)
    print(f"top1.avg:   ", top1.avg)

