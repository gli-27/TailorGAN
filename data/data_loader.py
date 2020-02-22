import random
import torch
import numpy as np
from skimage import io
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import pandas as pd
from util import util
from PIL import Image
import torchvision.utils as vutils

class CollarDataset(Dataset):
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
        # if torch.is_tensor(index):
        #    index = index.tolist()

        row = self.data_df.iloc[index]
        org_img = io.imread(row.tgt_imgPath)
        src_img = io.imread(row.src_imgPath)
        # edge_img = io.imread(row.part_edgePath)
        # FOR WO edge input, use RGB instead.
        edge_img = io.imread(row.tgt_imgPath)
        # refer_row = self.data_df.sample(n=1, replace=False, axis=0)
        idx = random.randint(0, len(self.data_df)-1)
        refer_row = self.data_df.iloc[idx]
        refer_edge_img = io.imread(refer_row.part_edgePath)

        # collar_type = self.type_list.copy()
        # collar_type[row.collar_type] = 1
        # collar_type = torch.Tensor(collar_type)
        org_collar_type = row.collar_type
        refer_collar_type = refer_row.collar_type
        refer_org_img = io.imread(refer_row.tgt_imgPath)

        src_img_tensor = self.org_transform(src_img.astype(np.uint8))
        orig_img_tensor = self.org_transform(np.uint8(org_img))
        refer_org_img_tensor = self.org_transform(np.uint8(refer_org_img))
        # gray_img_tensor = self.transform(np.uint8(gray_img))
        edge_tensor = self.transform(np.uint8(edge_img))
        refer_edge_tensor = self.transform(np.uint8(refer_edge_img))
        return [edge_tensor, refer_edge_tensor, src_img_tensor,
                org_collar_type, refer_collar_type, orig_img_tensor, refer_org_img_tensor]

    def __len__(self):
        return len(self.data_df)

class SleeveDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.data_df = pd.read_csv(opt.data_root + opt.data_path)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128), interpolation=2),
            transforms.RandomAffine(degrees=5),
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
        row = self.data_df.iloc[index]
        idx = random.randint(0, len(self.data_df) - 1)
        refer_row = self.data_df.iloc[idx]
        org_img = io.imread(row.imageName)
        edge_img = io.imread(row.CroppedSleeve)
        cropped_img = io.imread(row.cropped_img_path)
        refer_org_img = io.imread(refer_row.imageName)
        refer_edge_img = io.imread(refer_row.CroppedSleeve)

        org_img_tensor = self.org_transform(np.uint8(org_img))
        src_img_tensor = self.org_transform(cropped_img.astype(np.uint8))
        edge_tensor = self.transform(edge_img)

        refer_img_tensor = self.org_transform(np.uint8(refer_org_img))
        refer_edge_tensor = self.transform(refer_edge_img)

        org_sleeve_type = row.type
        refer_sleeve_type = refer_row.type

        return [edge_tensor, refer_edge_tensor, src_img_tensor,
                org_sleeve_type, refer_sleeve_type, org_img_tensor, refer_img_tensor]

    def __len__(self):
        return len(self.data_df)


class SleeveCrop(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.data_df = pd.read_csv(opt.data_root + opt.data_path)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128), interpolation=2),
            transforms.RandomAffine(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.org_transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((128, 128), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __getitem__(self, index):
        row = self.data_df.iloc[index]
        edge_img = io.imread(row.edge_path)

        org_h, org_w = row.orig_H, row.orig_W
        upper_x, upper_y = int(row.upper_X * 256 / org_w), int(row.upper_Y * 256 / org_h)
        upper_w, upper_h = int(row.upper_W * 256 / org_w), int(row.upper_H * 256 / org_h)
        shoulder1_x, shoulder1_y = int(row.shoulder1_x * 256 / org_w), int(row.shoulder1_y * 256 / org_h)
        shoulder2_x, shoulder2_y = int(row.shoulder2_x * 256 / org_w), int(row.shoulder2_y * 256 / org_h)
        sleeve1_x, sleeve1_y = int(row.sleeve1_x * 256 / org_w), int(row.sleeve1_y * 256 / org_h)
        sleeve2_x, sleeve2_y = int(row.sleeve2_x * 256 / org_w), int(row.sleeve2_y * 256 / org_h)

        mask = np.zeros((256, 256))
        mask[shoulder1_y:, sleeve1_x - 15:shoulder1_x + 15] = 1
        mask[shoulder2_y:, shoulder2_x - 15:sleeve2_x + 15] = 1

        croppedSleeve = edge_img*mask

        strlist = row.edge_path.split('/')
        croppedSleeveImgName = strlist[-1]

        edge_img = torch.Tensor(edge_img)
        edge_img = edge_img.repeat(3, 1, 1)
        croppedSleeve = torch.Tensor(croppedSleeve)
        croppedSleeve = croppedSleeve.repeat(3, 1, 1)

        path = './result/croppedSleeve/'
        util.mkdir(path)
        vutils.save_image(
            croppedSleeve.detach(), '%s/%s' % (path, croppedSleeveImgName),
            normalize=True
        )

        return 0

    def __len__(self):
        return len(self.data_df)

class CollarTest(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.collar_df = pd.read_csv(opt.data_root + '/collarTestSet.csv')

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128), interpolation=2),
            transforms.RandomAffine(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.org_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __getitem__(self, idx):
        refIdx = np.random.randint(0, len(self.collar_df)-1)
        baseRow = self.collar_df.iloc[idx]
        refRow = self.collar_df.iloc[refIdx]

        edge = io.imread(refRow.part_edgePath)
        src = io.imread(baseRow.src_imgPath)
        org = io.imread(baseRow.tgt_imgPath)
        refType = refRow.collar_type

        edgeTensor = self.transform(np.uint8(edge))
        srcTensor = self.org_transform(src.astype(np.uint8))
        orgTensor = self.org_transform(org.astype(np.uint8))

        return [edgeTensor, srcTensor, orgTensor, refType]

    def __len__(self):
        return len(self.collar_df)

class SleeveTestDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.shortSleeve_df = pd.read_csv(opt.data_root + '/shortSleeveSet.csv')
        self.longSleeve_df = pd.read_csv(opt.data_root + '/longSleeveSet.csv')

    def __getitem__(self, index):
        shortRow = self.shortSleeve_df.iloc[index]
        longRow = self.longSleeve_df.iloc[index]

        shortEdge = io.imread(shortRow.CroppedSleeve)
        shortSrc = io.imread(shortRow.cropped_img_path)
        shortEdgeTensor = self.transform(np.uint8(shortEdge))
        shortSrcTensor = self.org_transform(shortSrc.astype(np.uint8))

        longEdge = io.imread(longRow.CroppedSleeve)
        longSrc = io.imread(longRow.cropped_img_path)
        longEdgeTensor = self.transform(np.uint8(longEdge))
        longSrcTensor = self.org_transform(longSrc.astype(np.uint8))

        shortOrg = io.imread(shortRow.imageName)
        shortOrgTensor = self.org_transform(shortOrg.astype(np.uint8))
        longOrg = io.imread(longRow.imageName)
        longOrgTensor = self.org_transform(longOrg.astype(np.uint8))

        return [shortEdgeTensor, shortSrcTensor, longEdgeTensor, longSrcTensor, shortOrgTensor, longOrgTensor]

    def __len__(self):
        return min(len(self.shortSleeve_df), len(self.longSleeve_df))


class SleeveTest(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.sleeve_df = pd.read_csv(opt.data_root + '/test_sleeve.csv')

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128), interpolation=2),
            transforms.RandomAffine(degrees=5),
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
        refIdx = np.random.choice(len(self.sleeve_df), 1)[0]
        refRow = self.sleeve_df.iloc[refIdx]
        row = self.sleeve_df.iloc[index]

        refEdge = io.imread(refRow.CroppedSleeve)
        src = io.imread(row.cropped_img_path)
        org = io.imread(row.imageName)

        edgeTensor = self.transform(np.uint8(refEdge))
        srcTensor = self.org_transform(src.astype(np.uint8))
        orgTensor = self.org_transform(org.astype(np.uint8))

        return [edgeTensor, srcTensor, orgTensor]

    def __len__(self):
        return len(self.sleeve_df)
