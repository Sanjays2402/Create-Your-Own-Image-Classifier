import torch
import sys
from torch import nn
from load_images import load_images
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, models, transforms
from workspace_utils import active_session
import argparse

#parse input arguments
parser = argparse.ArgumentParser(description='Trains image classifier model')
parser.add_argument('data_dir', action="store")
parser.add_argument('--save_dir', action="store")
parser.add_argument('--arch', default='alexnet',action="store", dest="arch")
parser.add_argument('--learning_rate', default = 0.003, action="store", dest="learning_rate")
parser.add_argument('--epochs', default = 10, action="store", dest="epochs")
parser.add_argument('--hidden_units', default=512, action="store", dest="hidden_units")
parser.add_argument('--gpu', action="store_true", default=False, dest="boolean_gpu")
args = parser.parse_args()

train_dataset, validation_dataset, test_dataset, trainloader, validloader, testloader = load_images(args.data_dir)

#assign gpu
if args.boolean_gpu and torch.cuda.is_available() == True:
    device = torch.device("cuda")
    print("Running on GPU")
elif args.boolean_gpu == True:
    device = torch.device("cpu")
    print("GPU selected, but running on CPU as GPU is unavailable")   
else:
    device = torch.device("cpu")
    print("Running on CPU")

#assign architecture for model  
if args.arch =='alexnet':
    model = models.alexnet(pretrained=True)
elif args.arch == 'vgg16':
    model = models.vgg16(pretrained = True)
elif args.arch == 'vgg13':
    model = models.vgg13(pretrained = True)
elif args.arch == 'densenet121':
    model = models.densenet121(pretrained = True)
elif args.arch == 'densenet169':
    model = models.densenet169(pretrained = True)
else:
    print("Invalid model chosen; Choose between alexnet, vgg13, densenet121 and densenet169; Alexnet is the default model")
    sys.exit()
    
#freezing parameters of pretrained model
for param in model.parameters():
    param.requires_grad = False
    
input_size = {'vgg16':25088,'vgg13':25088,'alexnet':9216,'densenet121':1024, 'densenet169':1024}
model.classifier = nn.Sequential(nn.Linear(input_size[args.arch], args.hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(args.hidden_units, 102),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
model.to(device)

#training & validation 
with active_session():
    epochs = args.epochs
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            model.eval()
            validation_loss = 0
            accuracy = 0
            with torch.no_grad():
                for images, labels in validloader:
                    images, labels = images.to(device), labels.to(device)
                    logps = model.forward(images)
                    validation_loss += criterion(logps, labels)

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
            
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "train loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                  "validation loss: {:.3f}.. ".format(validation_loss/len(validloader)),
                  "validation accuracy: {:.3f}".format(accuracy/len(validloader)))
            running_loss = 0
            model.train()

#save checkpoint
model.class_to_idx = train_dataset.class_to_idx
checkpoint = {'input_size': input_size[args.arch],
              'output_size': 102,
              'hidden_layers': 1,
              'class_to_idx' : model.class_to_idx,
              'epochs' : epochs,
              'optimizer_state' : optimizer.state_dict(),
              'state_dict': model.classifier.state_dict(),  
              'classifier' : model.classifier,
              'arch' : args.arch}
torch.save(checkpoint, 'checkpoint.pth')