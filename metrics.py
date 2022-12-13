from models.model import *
from torch.utils.data import DataLoader
from dataset.dataset import *
from torch.utils.data import random_split
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torchvision import transforms, models
import numpy as np
import sys


if __name__=='__main__':

    input_transform = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    if sys.argv[1] == 'convNet':
        val_model = MalNet()
        PATH = 'models/convNet.pt'
        input_transform = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    elif sys.argv[1] == 'resNet18':
        val_model = models.resnet18(pretrained=True)
        val_model.fc = nn.Linear(512, 2, bias=True)
        PATH = 'models/resNet18.pt'
    elif sys.argv[1] == 'resNet50':
        val_model = models.resnet50(pretrained=True)
        val_model.fc = nn.Linear(2048, 2, bias=True)
        PATH = 'models/resNet50.pt'

    
    data = MalariaDataset(image_transforms=input_transform, label_transforms=None)
    length = data.__len__()
    classes = data.classes


    train_data, val_data, test_data = random_split(data, [19290, 5513, 2755])

    test_dataloader = DataLoader(test_data,batch_size=32, shuffle=False)


    val_model.load_state_dict(torch.load(PATH))

    val_model.eval()
    with torch.no_grad():
        y_pred = []
        y_true = []

        for i,data in enumerate(test_dataloader,0):

            inputs, labels = data
            outputs = val_model(inputs)
            _, predictons = torch.max(outputs,1)
            y_pred.extend(predictons)
            y_true.extend(labels)

    conf_mat = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(conf_mat/np.sum(conf_mat), index = [i for i in classes],
                     columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    plt.title("Confusion Matrix")
    sn.heatmap(df_cm, annot=True)
    plt.show()



    
