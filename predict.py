import torch
import sys
from torch import nn
from PIL import Image
import torch.nn.functional as F
from torch import optim
from process_image import process_image
from torchvision import models
from workspace_utils import active_session
import argparse
import json

#parse arguments
parser = argparse.ArgumentParser(description='Predicts flower name and probability using input image and checkpoint')
parser.add_argument('path_to_image', action="store")
parser.add_argument('checkpoint', action="store")
parser.add_argument('--top_k', action="store", default=1, type=int)
parser.add_argument('--category_names', action="store")
parser.add_argument('--gpu', action="store_true", default=False, dest="boolean_gpu")
args = parser.parse_args()
image_path = args.path_to_image
checkpoint_path = args.checkpoint

#assign GPU
if args.boolean_gpu and torch.cuda.is_available() == True:
    device = torch.device("cuda")
    print("Running on GPU")
elif args.boolean_gpu == True:
    device = torch.device("cpu")
    print("GPU selected, but running on CPU as GPU is unavailable")   
else:
    device = torch.device("cpu")
    print("Running on CPU")

#load checkpoint
checkpoint = torch.load(checkpoint_path)
model = getattr(models, checkpoint['arch'])(pretrained=True)
model.to(device)
model.classifier = checkpoint['classifier']
model.class_to_idx = checkpoint['class_to_idx']
model.classifier.load_state_dict(checkpoint['state_dict'])
epochs = checkpoint["epochs"]

optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
optimizer.load_state_dict(checkpoint['optimizer_state'])    

#predict flower name and its probability  
with active_session():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        model.eval()
        model.to(device)
        image = process_image(image_path)
        image = torch.FloatTensor(image).unsqueeze_(0)
        image = image.to(device)
        logps = model.forward(image)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(args.top_k, dim=1)

        top_p = top_p.cpu().detach().numpy().tolist()[0]
        top_class = top_class.cpu().detach().numpy().tolist()[0]

        #find the flower name from index
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        top_class_mapped = [idx_to_class[i] for i in top_class]
        
        #map category to names
        if args.category_names is not None:
            with open(args.category_names, 'r') as f:
                cat_to_name = json.load(f)
            flower = [cat_to_name[i] for i in top_class_mapped]
            for i in range(len(flower)):
                print("Flower:{}.. Probability:{:.3f}".format(flower[i],top_p[i]))
        else:
            for i in range(len(top_class_mapped)):
                print("Flower Category:{}.. Probability:{:.3f}".format(top_class_mapped[i],top_p[i]))
        
        model.train()