import os
from torchvision.io import read_image
from torch.utils.data import Dataset
from PIL import Image

#create data label_map
class MalariaDataset(Dataset):
    def __init__(self, data_dir='data', image_transforms=None, label_transforms=None):
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

        self.img_transform = image_transforms
        self.lbl_transform = label_transforms
        

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir,self.classes[self.label_list[idx]],self.image_list[idx])
        image = Image.open(img_path)
        label = self.label_list[idx]
        if self.img_transform:
            image = self.img_transform(image)
        if self.lbl_transform is not None:
            label = self.lbl_transform(label)
        return image, label

# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# label = train_labels[0]
# # print(img.numpy().transpose(1,2,0).shape)
# plt.imshow(img.numpy().transpose(1,2,0), cmap="gray")
# plt.show()
# print(f"Label: {classes[label]}")
