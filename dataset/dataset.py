import os
from torchvision.io import read_image
from torch.utils.data import Dataset


#create data label_map
class MalariaDataset(Dataset):
    def __init__(self, data_dir='data', transform=None):

        self.data_dir = data_dir
        files = os.listdir(data_dir)
        self.classes = [file 
        for file in files 
        if os.path.isdir(os.path.join(data_dir,file))]

        self.label_map = {cls : idx for idx , cls in enumerate(self.classes)}
        self.class_count = {cls : len(os.listdir(os.path.join(self.data_dir, cls))) for cls in self.label_map}

        self.image_list = [ path for cls in self.classes for path in os.listdir(os.path.join(data_dir, cls))]
        self.label_list = [label for cls in self.classes for label in [self.label_map[cls]]*self.class_count[cls]]
        self.img_label_dict = dict(zip(self.image_list, self.label_list))

        self.transform = transform
        

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir,self.classes[self.label_list[idx]],self.image_list[idx])
        image = read_image(img_path)
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

