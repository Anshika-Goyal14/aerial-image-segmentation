import os 
from PIL import Image 
from torch.utils.data import Dataset 
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torchvision import transforms 

class LandCoverDataset(Dataset):
    def __init__(self,  patch_names, output_dir, transform = None):
        self.patch_names = patch_names 
        self.transform = transform
        self.output_dir = output_dir
        
    def __len__(self):
        return len(self.patch_names)

    def __getitem__(self,idx):
        patch_name = self.patch_names[idx]
        img_path = os.path.join(self.output_dir, patch_name + ".jpg")
        mask_path = os.path.join(self.output_dir, patch_name + "_m.png")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") #converts to grayscale

        if self.transform: 
           image = self.transform(image)

        mask = torch.from_numpy(np.array(mask)).long() #gives a (512, 512) numpy array of uint8 integers and .long() cobverts to int64 so that its easier to calculate cross entropy loss
        return image, mask


def make_splits(txt_dir, output_dir ):
      def read_txt(path):
          with open(path, "r") as f :
              return [line.strip() for line in f.readlines() if line.strip()]
      train_patches = read_txt(os.path.join(txt_dir, "train.txt"))
      val_patches = read_txt(os.path.join(txt_dir, "val.txt"))
      test_patches = read_txt(os.path.join(txt_dir, "test.txt"))

      return train_patches, val_patches, test_patches 


if __name__ == "__main__":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.299, 0.224, 0.225))])

        train_patches, val_patches, test_patches = make_splits("landcover_ai", "landcover_ai/output")

        print(f"Train: {len(train_patches)}, Val: {len(val_patches)}, Test: {len(test_patches)}")

        dataset = LandCoverDataset(train_patches, "landcover_ai/output", transform=transform)
        image, mask = dataset[0]
        print(f"Image shape: {image.shape}")   
        print(f"Mask shape: {mask.shape}")     
        print(f"Mask unique values: {mask.unique()}")  
                                                                                                                                
      

      
      
           




