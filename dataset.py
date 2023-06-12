import os
import json

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from skimage.draw import polygon
from tqdm import tqdm

import matplotlib.pyplot as plt


class HuBMAP(Dataset):
    ''' Training dataset for the HuBMAP Kaggle Competition '''
    def __init__(self, data_dir=os.path.join('.', 'data')):
        self.data_dir = data_dir
        # Load in the training labels
        with open(os.path.join(data_dir, 'polygons.jsonl'), 'r') as polygons_file:
            polygons = list(polygons_file)
        self.polygons = [json.loads(p) for p in polygons]

        # Load all of the training images and annotations into memory
        self.img_size = 512
        self.images = []
        self.blood_vessel_masks = [] # target structure
        self.glomerulus_masks = [] # regions to avoid labeling (glomerulus)
        self.unsure_masks = [] # regions we are unsure about
        print("Loading in images and converting annotations to polygon masks...")
        for poly in tqdm(self.polygons):
            id = poly['id']
            # Get image using id
            image = Image.open(os.path.join(data_dir, 'train', f'{id}.tif'))
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
            blood_vessel_mask = self.coordinates_to_mask([item for sublist in blood_vessel_coords for item in sublist])
            glomerulus_mask = self.coordinates_to_mask([item for sublist in glomerulus_coords for item in sublist])
            unsure_mask = self.coordinates_to_mask([item for sublist in unsure_coords for item in sublist])
            self.blood_vessel_masks.append(blood_vessel_mask)
            self.glomerulus_masks.append(glomerulus_mask)
            self.unsure_masks.append(unsure_mask)
        print("Done.")
        # Set up image transformations
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512), antialias=True)
        ])

    def coordinates_to_mask(self, coordinates):
        if coordinates == None or coordinates == []:
            return torch.zeros((self.img_size, self.img_size), dtype=torch.bool)
        coords = torch.tensor(coordinates)
        mask = torch.zeros((self.img_size, self.img_size), dtype=torch.bool)
        mask[coords[:, 0], coords[:, 1]] = True
        return mask

    def __getitem__(self, i):
        # TODO: Remove glomerulus masks since they will be available in the test set
        # Instead, we should remove any annotations within the glomerulus mask
        # and ignore any predictions from our model in the region
        return {
            'image': self.transforms(self.images[i]).half(),
            'blood_vessel_mask': self.blood_vessel_masks[i].long(),
            'glomerulus_mask': self.glomerulus_masks[i].long(),
            'unsure_mask': self.unsure_masks[i].long(),
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
        print(f"Blood Vessel Shape: {d['blood_vessel_mask'].shape}")
        print(f"Blood Vessel Masks: {d['blood_vessel_mask']}")
        print(f"Glomerulus Shape: {d['glomerulus_mask'].shape}")
        print(f"Glomerulus Masks: {d['glomerulus_mask']}")
        print(f"Unsure Shape: {d['unsure_mask'].shape}")
        print(f"Unsure Masks: {d['unsure_mask']}")
        if i % 5 == 0:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
            ax1.imshow(torch.permute(d['image'][0], (1, 2, 0)))
            ax2.imshow(d['blood_vessel_mask'][0])
            ax3.imshow(d['glomerulus_mask'][0])
            ax4.imshow(d['unsure_mask'][0])
            plt.show()
            break
