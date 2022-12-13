from glob import glob
from PIL import Image
from termcolor import colored
from models.model import *
import numpy as np
import matplotlib.pyplot as plt
import sys

def load_input_image(img_path):
    image = Image.open(img_path)
    prediction_transform = transforms.Compose([transforms.Resize(size=(32, 32)),
                                     transforms.ToTensor(), 
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                          [0.229, 0.224, 0.225])])
    image = prediction_transform(image)[:3,:,:].unsqueeze(0)
    return image

def predict_malaria(model, class_names, img_path):
    img = load_input_image(img_path)
    model = model.cpu()
    model.eval()
    idx = torch.argmax(model(img))
    return class_names[idx]

def predict(model):
    class_names=['Parasitized','Uninfected']
    inf = np.array(glob("data/Parasitized/*"))
    uninf = np.array(glob("data/Uninfected/*"))
    for i in range(3):
        img_path=inf[i]
        img = Image.open(img_path)
        if predict_malaria(model, class_names, img_path) == 'Parasitized':
            print(colored('Parasitized', 'green'))
        else:
            print(colored('Uninfected', 'red'))
        plt.imshow(img)
        plt.show()
    for i in range(3):
        img_path=uninf[i]
        img = Image.open(img_path)
        if predict_malaria(model, class_names, img_path) == 'Uninfected':
            print(colored('Uninfected', 'green'))
        else:
            print(colored('Parasitized', 'red'))        
        plt.imshow(img)
        plt.show()
        pass

if __name__=='__main__':

    use_cuda = torch.cuda.is_available()
    if sys.argv[1] == 'convNet':
        test_model = MalNet()
        if use_cuda:
            test_model = test_model.cuda()
        PATH = 'models/convNet.pt'
        test_model.load_state_dict(torch.load(PATH))
        predict(model=test_model)
