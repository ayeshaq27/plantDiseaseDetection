"""
Custom dataset class that tells PyTorch:
  - how to find images
  - how to read them
  - how to apply transforms
  - how to return (image, label)
"""
from torch.utils.data import Dataset
from PIL import Image
import os

class PlantVillageDataset(Dataset):
  def __init__(self, root_dir, transform=None):
    self.root_dir = root_dir
    self.transform = transform
    self.classes = sorted(os.listdir(root_dir))

    self.image_paths = []
    self.labels = []

    for idx, cls in enumerate(self.classes):
      class_path = os.path.join(root_dir, cls)
      img_files = os.listdir(class_path)

      for img_name in img_files:
        self.image_paths.append(os.path.join(class_path, img_name))
        self.labels.append(idx)

  def __len__(self):
    return len(self.image_paths)
  
  def __getitem__(self,index):

    #Step 1 -> load image
    img_path = self.image_paths[index]
    img = Image.open(img_path).convert("RGB")

    #Step 2 -> apply transform
    if self.transform:
      img = self.transform(img)
    
    #Step 3 -> return (img, label)
    label = self.labels[index]
    return (img, label)