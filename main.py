from torch.utils.data import random_split
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset.dataset import *

def main():
    train_transform = transforms.Resize((125,140))
    data = MalariaDataset(transform=train_transform)
    length = data.__len__()
    train_data, val_data, test_data = random_split(data, [int(0.7*length), int(0.1*length), int(0.2*length)])

    train_dataloader = DataLoader(train_data,batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_data,batch_size=32, shuffle=True)

    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    print(img.numpy().transpose(1,2,0).shape)
    plt.imshow(img.numpy().transpose(1,2,0), cmap="gray")
    plt.show()
    print(f"Label: {label}")

if __name__=='__main__':
    main()