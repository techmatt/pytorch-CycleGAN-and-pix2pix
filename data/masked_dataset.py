import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image


class MaskedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase)

        self.A_paths = sorted(make_dataset(self.dir_A))

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

        self.yMaskCutoff = 50
        self.maskYIndices = [i for i in range(0, opt.fineSize)]
        for i in range(0, self.yMaskCutoff):
            self.maskYIndices[i] = self.yMaskCutoff * 2 - i
        self.maskYIndices = torch.LongTensor(self.maskYIndices)

        self.maskTensor = torch.Tensor(self.opt.fineSize, self.opt.fineSize)
        self.maskTensor.fill_(0.0)
        for i in range(0, self.yMaskCutoff):
            self.maskTensor[i,:].fill_(0.5 + 0.5 * i / self.yMaskCutoff)


    def applyRandomMask(self, A):
        #self.A = self.Tensor(opt.input_nc, opt.fineSize, opt.fineSize)

        result = torch.Tensor(4, self.opt.fineSize, self.opt.fineSize)

        #print(self.maskYIndices)

        #fixedA = A.index_select(1, self.maskYIndices)
        #print(fixedA.size())

        result[0:3, :, :] = A.index_select(1, self.maskYIndices)
        result[3, :, :].copy_(self.maskTensor)

        #result[0:3, :, :].copy_(A)
        #result[3, :, :].fill_(0.0)

        #maskW = random.randint(self.opt.mask_rect_min_size, self.opt.mask_rect_max_size)
        #maskH = random.randint(self.opt.mask_rect_min_size, self.opt.mask_rect_max_size)

        #w = A.size(2)
        #h = A.size(1)
        #w_offset = random.randint(0, max(0, w - maskW - 1))
        #h_offset = random.randint(0, max(0, h - maskH - 1))

        #maskTensor = result[:, h_offset:h_offset + maskH, w_offset:w_offset + maskW]
        #maskTensor.fill_(1.0)

        return result


    def __getitem__(self, index):
        A_path = self.A_paths[index]
        ARaw = Image.open(A_path).convert('RGB')

        squareDim = min(ARaw.size[0] - 1, ARaw.size[1] - 1)
        xStart = random.randint(0, ARaw.size[0] - squareDim)
        yStart = random.randint(0, ARaw.size[1] - squareDim)
        ARaw = ARaw.crop((xStart, yStart, xStart + squareDim, yStart + squareDim))

        ARaw = ARaw.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        ARaw = self.transform(ARaw)

        #crop in half because this dataset is AB
        #ARaw = ARaw[:, :, 0:ARaw.size(2)/2]

        w = ARaw.size(2)
        h = ARaw.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = ARaw[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        B = A.clone()

        A = self.applyRandomMask(A)

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': A_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'MaskedDataset'
