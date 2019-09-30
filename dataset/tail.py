import os
import cv2
from PIL import Image

import matplotlib.pyplot as plt
import torch
import torch.nn.functional
import numpy as np
import torch.utils.data
import torchvision


class Tail(torch.utils.data.Dataset):
    def __init__(self, label_path, day_or_night='night', distance=30, task='break', train=True, car_type='',
                 if_sever=False, transform=True):
        self.pair = []
        self.dataset = []
        self.transform = transform
        if train:
            if if_sever:
                self.label_path = label_path + '/' + str(
                    distance) + '_' + day_or_night + '_sever_train.txt'
            else:
                self.label_path = label_path + '\\' + str(
                    distance) + '_' + day_or_night + '_train.txt'
            f = open(self.label_path)
        else:
            if if_sever:
                self.label_path = label_path + '/' + str(
                    distance) + '_' + day_or_night + '_sever_test.txt'
            else:
                # self.label_path='D:\\light_state_recognition_data_20190909\\tail\\over50_night_test.txt'
                self.label_path = label_path + '\\' + str(
                    distance) + '_' + day_or_night + '_test.txt'
            f = open(self.label_path)
        print("Loading label: ", self.label_path)
        all_data = f.readlines()
        for data in all_data:
            if day_or_night in data:
                if car_type in data:
                    data_list = data.split(',')
                    self.pair.append(data_list[0])
                    if task == 'break':
                        self.pair.append(data_list[5])
                    elif task == 'turn':
                        self.pair.append(data_list[6])
                    self.pair.append(data_list[1:5])
                    self.dataset.append(self.pair)
                    self.pair = []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_current = self.dataset[index]
        img_path = data_current[0]
        img = Image.open(img_path)
        origin = cv2.imread(img_path)
        region = img.crop(
            (int(data_current[2][0])-10, int(data_current[2][1])-10, int(int(data_current[2][0]) + int(data_current[2][2])+10),
             int(int(data_current[2][1]) + int(data_current[2][3])+10)
             ))
        # region=np.transpose(region, (2, 0, 1)).astype(np.float32)
        if self.transform == True:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((125,125)),
                torchvision.transforms.RandomCrop(112),
                torchvision.transforms.ColorJitter(brightness=0.5, contrast=0, hue=0),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        sample = dict()
        sample["img"] = transform(region)
        sample['region'] = data_current[2]

        sample['origin'] = origin
        if data_current[1] == '1':
            # sample["target"] = torch.Tensor([1, 0])
            sample["target"] = torch.Tensor([0])
        else:
            # sample["target"] = torch.Tensor([0, 1])
            sample["target"] = torch.Tensor([1])
        sample["gt"] = data_current[1]
        return sample


if __name__ == '__main__':
    dataset = Tail("D:\\light_state_recognition_data_20190909\\tail", day_or_night='day', car_type='', train=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    for i, sample in enumerate(data_loader):
        # origin = sample['origin'].numpy().astype(np.uint8)
        print(sample['gt'], sample['target'])
        # cv2.imshow('img', origin[0])
        to_img = torchvision.transforms.ToPILImage()
        img = to_img(sample['img'][0])
        img.show()

        # cv2.imshow('img', np.transpose(img[0], (1, 2, 0)))
        # cv2.waitKey()
        while i > 10:
            break
