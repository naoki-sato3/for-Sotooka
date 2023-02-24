'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import wandb

import optimizers as fw
import constraints as c

from models import *
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--optimizer',default="adam", type=str, help='[mome,sgd,adam,rms,adag,adaw,amsg,cgad,cgam,sfw]')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
#net = SimpleDLA()
#net = ResNet50()
net = net.to(device)
if device == 'cuda:0':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

def func(steps):
    return 1/(steps+1)

def func2(steps):
    b = steps + 1
    b = b ** 0.5
    return 1/b

criterion = nn.CrossEntropyLoss()
if args.optimizer == "mome":
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)               #Momentum
elif args.optimizer == "adam":
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))                           #Adam
elif args.optimizer == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.0)                                  #SGD
elif args.optimizer == "rms":
    optimizer = optim.RMSprop(net.parameters(), lr=0.05, alpha=0.9)                                 #RMSProp      
elif args.optimizer == "adag":
    optimizer = optim.Adagrad(net.parameters(), lr=0.01, weight_decay=0)                            #AdaGrad      
elif args.optimizer == "adaw":
    optimizer = optim.AdamW(net.parameters(), lr=0.01, betas=(0.9, 0.999))                          #AdamW        
elif args.optimizer == "amsg":
    optimizer = optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), amsgrad=True)             #AMSGrad       
elif args.optimizer == "sfw":
    optimizer = fw.SFW(net.parameters(), lr=0.01, momentum=0.0)                                     #SFW   

#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)                        #cosin
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=func)                           #diminishing1   
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=func2)                            #diminishing2

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        if args.optimizer == "sfw":
            optimizer.step(constraints=const)
        else:
            optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    training_acc = 100.*correct/total
    wandb.log({'training_acc': training_acc,
               'training_loss': train_loss/(batch_idx+1)})


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    wandb.log({'accuracy': acc})
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


wandb_project_name = "izumi-cg"
wandb_exp_name = f"{args.lr},"
wandb.init(config = args,
           project = wandb_project_name,
           name = wandb_exp_name,
           entity = "naoki-sato")
wandb.init(settings=wandb.Settings(start_method='fork'))

print(optimizer)
if args.optimizer == "sfw":
    const = c.create_lp_constraints(net, value=1e+10)
    c.print_constraints(net, const)

for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    last_lr = scheduler.get_last_lr()[0]
    wandb.log({'last_lr': last_lr})
    scheduler.step()
