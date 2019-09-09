import os
import torch
import numpy as np
from skimage import io
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import pandas as pd

class FashionDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.data_df = pd.read_csv(opt.data_root + opt.data_path)
        if self.opt.type_classifier == 'collar':
            self.type_list = [0] * self.opt.num_collar
            torch.Tensor(self.type_list)
        else:
            self.type_list = [0] * self.opt.num_sleeve
            torch.Tensor(self.type_list)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128), interpolation=2),
            transforms.RandomAffine(degrees=15, scale=(0.85, 1.55), translate=(0.10, 0.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.org_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        row = self.data_df.iloc[index]
        org_img = io.imread(row.tgt_imgPath)
        src_img = io.imread(row.src_imgPath)
        edge_img = io.imread(row.edge_imgPath)


        # imgInd = np.random.choice(len(self.data_df), 1)[0]
        # row_df = self.data_df.iloc[imgInd]
        # org_imgPath = row_df.tgt_imgPath
        # org_img = io.imread(org_imgPath)

        # edge_imgPath = row_df.edge_imgPath
        # edge_img = io.imread(edge_imgPath)

        gray_img = 0.2125*org_img[:, :, 0] + 0.7154*org_img[:, :, 1] + 0.0721*org_img[:, :, 2] # 255.0*rgb2gray(orig_img)

        gray_img = np.repeat(gray_img[:, :, np.newaxis], 3, axis=2)
        bbox_x1, bbox_y1 = row.landmark1_x, row.landmark1_y
        bbox_x2, bbox_y2 = row.landmark2_x, row.landmark2_y
        centX, centY = int((bbox_x1 + bbox_x2) / 2.0), int((bbox_y1 + bbox_y2) / 2.0)
        mask = np.zeros((256, 256, 3))

        tY2, bY2 = max(centY - 40, 0), min(256, centY + 40)
        tX2, bX2 = max(centX - 40, 0), min(256, centX + 40)
        tY, tX = max(centY - 64, 0), max(0, centX - 64)
        bY, bX = min(tY + 128, 256), min(tX + 128, 256)
        tY, tX = max(bY - 128, 0), max(0, bX - 128)
        mask[tY2:bY2, tX2:bX2, :] = 1.0

        part_edge_img = edge_img[tY:bY, tX:bX, :].copy()
        src_img = np.multiply(1.0 - mask, org_img)  # np.multiply(mask,blur_img)

        # collar_type = self.type_list.copy()
        # collar_type[row.collar_type] = 1
        # collar_type = torch.Tensor(collar_type)
        collar_type = row.collar_type

        src_img_tensor = self.org_transform(src_img.astype(np.uint8))
        orig_img_tensor = self.org_transform(np.uint8(org_img))
        gray_img_tensor = self.transform(np.uint8(gray_img))
        part_edge_tensor = self.transform(np.uint8(part_edge_img))
        return [gray_img_tensor, src_img_tensor, collar_type, orig_img_tensor]

    def __len__(self):
        return len(self.data_df)

