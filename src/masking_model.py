import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np

class CracksDataset(Dataset):
    ##'Characterizes a dataset for PyTorch'
    def __init__(self, data_path, masks_path):
        ##'Initialization'
        self.data_path = data_path
        self.masks_path = masks_path

        assert os.path.exists(self.data_path)
        assert os.path.exists(self.masks_path)

        self.data_len = len(os.listdir(self.data_path))
        assert self.data_len == len(os.listdir(self.masks_path))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        # Select sample
        data_file = os.listdir(self.data_path)[index]
        mask_file = os.listdir(self.masks_path)[index]

        # Load data and get label
        image = Image.open(os.path.join(self.data_path, data_file))
        mask = Image.open(os.path.join(self.masks_path, mask_file))

        to_tensor = transforms.ToTensor()

        image = to_tensor(image)
        image = image[0].unsqueeze(0)  # remove transparency channel.
        mask = to_tensor(mask)

        assert image.shape == mask.shape

        return image.cuda(), mask.cuda()

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        '''
        in_channels, 
        out_channels, 
        kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'
        '''
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=3),
        #     nn.ReLU(True),
        #     nn.Conv2d(16, 32, kernel_size=5),
        #     nn.ReLU(True))
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(32, 16, kernel_size=5),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(16, 1, kernel_size=3),
        #     nn.ReLU(True))

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=3),
            nn.ReLU(True))


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_encoderDecoder():
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Normalize((0.4914, 0.4822, 0.4466), (0.247, 0.243, 0.261))])
    #
    # trainTransform = tv.transforms.Compose([transforms.ToTensor(),
    #                                         transforms.Normalize((0.4914, 0.4822, 0.4466), (0.247, 0.243, 0.261))])

    if not torch.cuda.is_available():
        raise NotImplementedError

    trainset = CracksDataset(data_path='J:\Celebs_dataset\celeba_cracked\data',
                             masks_path='J:\Celebs_dataset\celeba_cracked\label')

    dataloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=0)

    num_epochs = 200

    model = Autoencoder()
    model.cuda()
    distance = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)

    for epoch in range(num_epochs):
        for data in tqdm(dataloader):
            img, mask = data
            img = Variable(img)
            # ===================forward=====================
            output = model(img)
            loss = distance(output, mask)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.data))
        show_pictures_in_batch(img, mask, output)

def _helper_show_pictures_in_batch(data):
    pic1 = data.detach().cpu().numpy()[1, 0, :]
    pic2 = data.detach().cpu().numpy()[2, 0, :]
    pic3 = data.detach().cpu().numpy()[3, 0, :]
    pic4 = data.detach().cpu().numpy()[4, 0, :]

    col1 = np.concatenate([pic1, pic2])
    col2 = np.concatenate([pic3, pic4])
    pic = np.concatenate([col1, col2], axis=1)

    return pic

def show_pictures_in_batch(data, masks, outputs):
    data = _helper_show_pictures_in_batch(data)
    masks = _helper_show_pictures_in_batch(masks)
    outputs = _helper_show_pictures_in_batch(outputs)

    pic = np.concatenate([data, masks, outputs])

    plt.imshow(pic)
    plt.ion()
    plt.show()
    plt.pause(0.001)

train_encoderDecoder()