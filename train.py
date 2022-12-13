from torch.utils.data import random_split
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn, optim
from dataset.dataset import *
from models.model import *
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time



def eval(model, valid_loader, use_cuda, criterion):

    model.eval()
    total = 0.0
    correct = 0.0
    valid_loss = 0.0
    for batch_id ,data in enumerate(valid_loader):
        inputs, labels = data
        if use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        valid_loss += ((1 / (batch_id + 1)) * (loss.data - valid_loss))
        pred = outputs.data.max(1, keepdim=True)[1]
        correct += np.sum(np.squeeze(pred.eq(labels.data.view_as(pred))).cpu().numpy())
        total += inputs.size(0)
    valid_acc = correct/total

    return valid_loss, valid_acc




def train(n_epochs, model, train_dataloader, optimizer, criterion, use_cuda, model_path, valid_loader):

    writer = SummaryWriter("runs/MalNet_"+str(time.time()))
    for epoch in range(n_epochs):

        model.train()
        running_loss = 0.0
        train_loss = 0.0
        train_acc = 0.0
        total = 0.0
        correct = 0.0
        for batch_id, data in tqdm(enumerate(train_dataloader,0)):

            inputs, labels = data
            if use_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels)

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            train_loss += ((1 / (batch_id + 1)) * (loss.data - train_loss))
            pred = outputs.data.max(1, keepdim=True)[1]
            correct += np.sum(np.squeeze(pred.eq(labels.data.view_as(pred))).cpu().numpy())
            total += inputs.size(0)
        
        
        train_acc = correct/total

        val_loss, val_acc = eval(model,valid_loader, use_cuda,criterion)

        print('Epoch: {}'.format(
            epoch+1
            ))
        print('Train Loss: {:.6f} \tVal Loss: {:.6f}'.format( 
            train_loss,
            val_loss
            ))
        print('Train Acc: {:.6f} \tValidation Acc: {:.6f}'.format( 
            train_acc,
            val_acc
            ))
        # print("training loss: ", train_loss.item(), "training acc: ", train_acc)
        # print("validation loss: ", val_loss.item(), "val acc: ", val_acc)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)

    #save the model
    torch.save(model.state_dict(), model_path)


def main():

    input_transform = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    label_transform = transforms.ToTensor()
    data = MalariaDataset(image_transforms=input_transform, label_transforms=None)
    length = data.__len__()
    classes = data.classes


    train_data, val_data, test_data = random_split(data, [19290, 5513, 2755])


    train_dataloader = DataLoader(train_data,batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=32, shuffle=True)


    net = MalNet()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001 , momentum=0.9)
    
    train(n_epochs=30, model=net, train_dataloader=train_dataloader, optimizer=optimizer, criterion=criterion,use_cuda=use_cuda, model_path='models/model.pt',valid_loader=val_dataloader)

    



    # writer.close()
if __name__=='__main__':
    main()