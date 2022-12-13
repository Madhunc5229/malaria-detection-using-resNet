from torch.utils.data import random_split
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn, optim
from dataset.dataset import *
from models.model import *


def main():
    input_transform = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    label_transform = transforms.ToTensor()
    data = MalariaDataset(image_transforms=input_transform, label_transforms=None)
    length = data.__len__()
    classes = data.classes
    print(classes)
    train_data, val_data, test_data = random_split(data, [int(0.7*length), int(0.2*length), int(0.1*length)])

    train_dataloader = DataLoader(train_data,batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_data,batch_size=32, shuffle=False)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001 , momentum=0.9)
    
   
if __name__=='__main__':
    main()