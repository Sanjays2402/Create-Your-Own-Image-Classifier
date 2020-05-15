import numpy as np
from PIL import Image

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    width, height = im.size
    new_width = 0
    new_height = 0
    if width <= height:
        new_width = 256
        new_height = int((height/width)* new_width)
    elif height < width:
        new_height = 256
        new_width = int((width/height)*new_height)
    size = new_width, new_height
    im = im.resize(size)

    left = (new_width - 224)/2
    top = (new_height - 224)/2
    right = (new_width + 224)/2
    bottom = (new_height + 224)/2
    im = im.crop((left, top, right, bottom))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = np.array(im)
    np_image = np_image/255
    np_image = (np_image - mean)/std
    
    return np_image.transpose(2,0,1)