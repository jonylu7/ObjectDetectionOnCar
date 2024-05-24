# generate by ChatGPT and modify by jonylu7

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os


class BoundingBoxDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")

        # Get bounding box coordinates
        boxes = self.annotations.iloc[idx, 1:5].values
        boxes = boxes.astype('float').reshape(-1, 4)

        if self.transform:
            image = self.transform(image)

        # Convert bounding box to tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)

        # Creating a dictionary to store the image and bounding boxes
        # sample = {'image': image, 'boxes': boxes}

        return (image, boxes)


if __name__ == '__main__':
    # Example usage
    from torchvision import transforms

    # Define transformations (if any)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Initialize dataset
    dataset = BoundingBoxDataset(csv_file='data/annotations.csv', root_dir='data/images/', transform=transform)

    ## seperate train and test
    annotations = pd.read_csv('data/annotations.csv')
    ##train_annotations, test_annotations = train_test_split(annotations, test_size=0.2, random_state=42)
    ##train_annotations.to_csv('data/train_annotations.csv', index=False)
    ##test_annotations.to_csv('data/test_annotations.csv', index=False)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=4)

    # Iterate through the DataLoader
    # or image, box in dataloader:
    # print(image)
    # print(box)
    # print(image.shape, box.shape)
    # print("\n")
