import argparse
import json
import logging
import os
import sagemaker_containers
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import numpy as np
from six import BytesIO

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


# Define Model Architectures -----------------------------------------------------------------------------------


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

class VGG(nn.Module):

    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    
    
    
# Define core training functions ---------------------------------------------------------------------------    
    
def calc_accuracy(model, data):
    size = 0.0
    correct = 0.0
    
    inputs, labels = data
    for j,output in enumerate(model(inputs)):
        if labels[j] == torch.max(output,0)[1]:
            correct += 1
        size += 1
    
    return (correct/size)
    
    
    
def _train(args):
    is_distributed = len(args.hosts) > 1 and args.dist_backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))

    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ['RANK'] = str(host_rank)
        dist.init_process_group(backend=args.dist_backend, rank=host_rank, world_size=world_size)
        logger.info(
            'Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
                args.dist_backend,
                dist.get_world_size()) + 'Current host rank is {}. Using cuda: {}. Number of gpus: {}'.format(
                dist.get_rank(), torch.cuda.is_available(), args.num_gpus))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info("Device Type: {}".format(device))

    logger.info("Loading Cifar10 dataset")
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    print("-----------------------------------", args.data_dir, "---------------------------------------")
    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True,
                                            download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers)
    valset = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers)
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False,
                                           download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers)

                
    logger.info("Model loaded")
    model = VGG('VGG11')
    print(model)
    training_log = open(os.path.join(args.model_dir, 'training_log.txt'), 'w+')
    if torch.cuda.device_count() > 1:
        logger.info("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    
    chck = 0
    n = 100
    print("-----------------------------Training has started--------------------------------------")
    for epoch in range(0, args.epochs):
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            curr_acc = calc_accuracy(model, data)
            running_acc += curr_acc
            if i % n == n-1:
                print('Epoch: %d, iter: %5d, loss: %.3f, acc: %.3f' %
                      (epoch + 1, i + 1, running_loss / n, running_acc / n))
                str_wr = 'Epoch: %d,iter: %5d,loss: %.3f,acc: %.3f\n' % (epoch + 1, i + 1, running_loss / n, running_acc / n)
                training_log.write(str_wr)
                
                running_loss = 0.0
                running_acc = 0.0
                
            if i % 400 == 399:
                break
                
        val_running_loss = 0.0
        val_running_acc = 0.0
        for j, val_data in enumerate(train_loader):
            if j > 400:
                val_running_acc += calc_accuracy(model,val_data)
                val_set_inputs, val_set_labels = val_data
                val_inputs,val_labels = val_set_inputs.to(device), val_set_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)
                val_running_loss += val_loss.item()
                
        print('Epoch: %d, Validation loss: %.3f, Val Accuracy: %.3f' %
                      (epoch + 1, val_running_loss / 100, val_running_acc / 100))
        str_wr = 'Epoch: %d,V: %.3f,VAcc: %.3f\n' % (epoch + 1, val_running_loss / 100, val_running_acc /100)
        training_log.write(str_wr)
    
    print('Finished Training')
    return _save_model(model, args.model_dir, 'final.pth')



def _save_model(model, model_dir, model_name):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, model_name)
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)

    
    
# Required Functions for Sagemaker Utility ----------------------------------------------------------------------    
    
    
    
def model_fn(model_dir):
    logger.info('model_fn')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGG('VGG11')
    if torch.cuda.device_count() > 1:
        logger.info("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

#     with open(os.path.join(model_dir, 'final.pth'), 'rb') as f:
#         model.load_state_dict(torch.load(f))
    print("Loading")
    
    pretrained_dict = torch.load(os.path.join(model_dir, 'final.pth'))
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)
    
    # model.load_state_dict(torch.load(os.path.join(model_dir, 'final.pth')))
    return model.to(device)



def input_fn(request_body, request_content_type):
    """An input_fn that loads a pickled tensor"""
    if request_content_type == 'application/x-npy':
        print ("Successful Data Recog!!!")
        return np.frombuffer(request_body,dtype=np.uint8).reshape((3,32,32))
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.
        pass

    
    
def predict_fn(input_data, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    transform_pred = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    with torch.no_grad():
        input_data_norm = transform_pred(torch.from_numpy(input_data))
        input_data_norm.unsqueeze_(0)
        print("Prediction Into Model")
        return model(input_data_norm.to(device))

    
    
def output_fn(prediction, content_type):
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    label = torch.max(prediction,1)[1].item()
    return classes[label]


# Define actions when script is called --------------------------------------------------------------------------------



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--workers', type=int, default=2, metavar='W',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', type=int, default=35, metavar='E',
                        help='number of total epochs to run (default: 50)')
    parser.add_argument('--batch_size', type=int, default=100, metavar='BS',
                        help='batch size (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--dist_backend', type=str, default='gloo', help='distributed backend (default: gloo)')

    env = sagemaker_containers.training_env()
    parser.add_argument('--hosts', type=list, default=env.hosts)
    parser.add_argument('--current-host', type=str, default=env.current_host)
    parser.add_argument('--model-dir', type=str, default=env.model_dir)
    parser.add_argument('--data-dir', type=str, default=env.channel_input_dirs.get('training'))
    parser.add_argument('--num-gpus', type=int, default=env.num_gpus)

    _train(parser.parse_args())