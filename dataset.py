import os
import json

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from skimage.draw import polygon
from tqdm import tqdm

import matplotlib.pyplot as plt

from utils import *


class HuBMAP(Dataset):
    ''' Dataset for the HuBMAP Kaggle Competition '''
    def __init__(self, data_dir=os.path.join('.', 'data'), submission=False, include_unsure=False):
        self.data_dir = data_dir
        self.submission = submission
        self.image_ids = []
        self.images = []
        self.masks = [] # target structure

        if not submission:
            # Load in the training labels
            with open(os.path.join(data_dir, 'polygons.jsonl'), 'r') as polygons_file:
                polygons = list(polygons_file)
            self.polygons = [json.loads(p) for p in polygons]

            # Load all of the training images and annotations into memory
            self.img_size = 512
            print("Loading in images and converting annotations to polygon masks...")
            for poly in tqdm(self.polygons):
                id = poly['id']
                # Get image using id
                image = Image.open(os.path.join(data_dir, 'train', f'{id}.tif'))
                self.image_ids.append(id)
                self.images.append(image)

                # Get all of the different annotations for this image
                ## A single image can have multiple annotations for each type
                blood_vessel_coords = []
                glomerulus_coords = []
                unsure_coords = []
                ## Load in the type and coordinates for each annotation
                annotations = poly['annotations']
                for ann in annotations:
                    type = ann['type']
                    assert len(ann['coordinates']) <= 1
                    coordinates = ann['coordinates'][0]
                    row_indices = [c[0] for c in coordinates] 
                    col_indices = [c[1] for c in coordinates]
                    row_indices, col_indices = polygon(row_indices, col_indices, (self.img_size, self.img_size))
                    coordinates = list(zip(row_indices, col_indices))
                    if type == 'blood_vessel':
                        blood_vessel_coords.append(coordinates)
                    elif type == 'glomerulus':
                        glomerulus_coords.append(coordinates)
                    else:
                        unsure_coords.append(coordinates) 
                mask = self.coordinates_to_mask([item for sublist in glomerulus_coords for item in sublist])
                if include_unsure and unsure_coords is not None and len(unsure_coords) > 0:
                    uns_coords = torch.tensor([item for sublist in unsure_coords for item in sublist])
                    mask[uns_coords[:, 1], uns_coords[:, 0]] = 2
                if blood_vessel_coords is not None and len(blood_vessel_coords) > 0:
                    bv_coords = torch.tensor([item for sublist in blood_vessel_coords for item in sublist])
                    # coordinates are (x, y) like a grid
                    mask[bv_coords[:, 1], bv_coords[:, 0]] = 2
                self.masks.append(mask)
            print("Done.")
        else:
            for image_file in os.listdir(os.path.join(data_dir, 'test')):
                image = Image.open(os.path.join(data_dir, 'test', image_file))
                id = image_file.split('.')[0]
                self.image_ids.append(id)
                self.images.append(image)
                self.masks.append(torch.zeros((512, 512)))

        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512), antialias=True)
        ])

    def transform(self, image, mask):
        image = F.to_tensor(image)
        resize = transforms.Resize(size=(512, 512), antialias=True)
        image = resize(image)
        mask = mask.unsqueeze(0)

        # horizontal flip
        if torch.rand(1).item() > 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)

        # vertical flip
        if torch.rand(1).item() > 0.5:
            image = F.vflip(image)
            mask = F.vflip(mask)

        # random elastic
        alpha = 50.0
        sigma = 5.0
        elastic_displacement = transforms.ElasticTransform.get_params([alpha, alpha], [sigma, sigma], [512, 512])
        image = F.elastic_transform(image, elastic_displacement)
        mask = F.elastic_transform(mask, elastic_displacement)

        return image, mask.squeeze()

    def coordinates_to_mask(self, coordinates):
        if coordinates is None or coordinates == []:
            return torch.zeros((self.img_size, self.img_size), dtype=torch.long)
        coords = torch.tensor(coordinates)
        mask = torch.zeros((self.img_size, self.img_size), dtype=torch.bool)
        # coordinates are (x, y) like a grid
        mask[coords[:, 1], coords[:, 0]] = True
        return mask.type(torch.long)

    def __getitem__(self, i):
        if self.submission:
            image = self.test_transforms(self.images[i])
            mask = self.masks[i]
            transformed_image = image
            transformed_mask = mask
        else:
            transformed_image, transformed_mask = self.transform(self.images[i], self.masks[i])
            image = self.test_transforms(self.images[i])
            mask = self.masks[i]
        return {
            'id': self.image_ids[i],
            'image': image,
            'mask': mask,
            'transformed_image': transformed_image,
            'transformed_mask': transformed_mask.type(torch.long),
        }

    def __len__(self):
        return len(self.images)
    

if __name__ == '__main__':
    data = HuBMAP()
    data_loader = DataLoader(data, batch_size=10, shuffle=True)
    print(f'Dataset Size: {len(data)}')
    for i, d in enumerate(data_loader):
        print(f"Idx: {i}")
        print(f"Image Shape: {d['image'].shape}")
        print(f"Image: {d['image']}")
        print(f"Blood Vessel Shape: {d['mask'].shape}")
        print(f"Blood Vessel Masks: {d['mask']}")
        fig, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(torch.permute(d['image'][0], (1, 2, 0)))
        ax[1, 0].imshow(torch.permute(d['transformed_image'][0], (1, 2, 0)))
        ax[0, 1].imshow(d['mask'][0])
        ax[1, 1].imshow(d['transformed_mask'][0])
        plt.show()
