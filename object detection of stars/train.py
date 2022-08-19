import typing as t
from typing import NamedTuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import DEVICE, synthesize_data
import math
import os
import torch.optim.lr_scheduler as lr_scheduler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import random



from utils import DEVICE, score_iou, synthesize_data

# set up the same seed make the experiment can be reproduce
def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    torch.cuda.manual_seed(seed)  # 为当前GPU设置
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    torch.backends.cudnn.benchmark = True  # 加快卷积运算
    torch.backends.cudnn.deterministic = True  # 固定网络结构


same_seeds(1)
import torch
import torch.nn as nn
import torch.nn.functional as F


class StarModel(nn.Module):
    '''
    Add the batchnormalization based on the vgg16 to solve problem of too slow gradient descent caused by different data scales
    '''
    def __init__(self):
        super(StarModel, self).__init__()

        self.conv1_1 = nn.Conv2d(1, 32, 3)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.maxpool1 = nn.MaxPool2d((2, 2), padding=(1, 1))
        self.norm0 = nn.BatchNorm2d(32, affine=False)

        self.conv2_1 = nn.Conv2d(32, 64, 3)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))
        self.maxpool2 = nn.MaxPool2d((2, 2), padding=(1, 1))
        self.norm1 = nn.BatchNorm2d(64, affine=False)

        self.conv3_1 = nn.Conv2d(64, 128, 3)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=(1, 1))
        self.conv3_3 = nn.Conv2d(128, 128, 3, padding=(1, 1))
        self.maxpool3 = nn.MaxPool2d((2, 2), padding=(1, 1))
        self.norm2 = nn.BatchNorm2d(128, affine=False)

        self.conv4_1 = nn.Conv2d(128, 256, 3)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=(1, 1))
        self.conv4_3 = nn.Conv2d(256, 256, 3, padding=(1, 1))
        self.maxpool4 = nn.MaxPool2d((2, 2), padding=(1, 1))
        self.norm3 = nn.BatchNorm2d(256, affine=False)

        self.conv5_1 = nn.Conv2d(256, 256, 3)
        self.conv5_2 = nn.Conv2d(256, 256, 3, padding=(1, 1))
        self.conv5_3 = nn.Conv2d(256, 256, 3, padding=(1, 1))
        self.maxpool5 = nn.MaxPool2d((2, 2), padding=(1, 1))
        self.norm4 = nn.BatchNorm2d(256, affine=False)
        # view

        self.fc1 = nn.Linear(9216,512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 5)


    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1_1(x)  # 222
        out = self.norm0(out)
        out = F.relu(out)
        out = self.conv1_2(out)  # 222

        out = F.relu(out)
        out = self.maxpool1(out)  # 112

        out = self.conv2_1(out)  # 110
        out = self.norm1(out)
        out = F.relu(out)
        out = self.conv2_2(out)  # 110

        out = F.relu(out)
        out = self.maxpool2(out)  # 56

        out = self.conv3_1(out)  # 54
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv3_2(out)  # 54

        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv3_3(out)  # 54

        out = F.relu(out)
        out = self.maxpool3(out)  # 28

        out = self.conv4_1(out)  # 26
        out = self.norm3(out)
        out = F.relu(out)
        out = self.conv4_2(out)  # 26

        out = F.relu(out)
        out = self.conv4_3(out)  # 26
        out = self.norm3(out)
        out = F.relu(out)
        out = self.maxpool4(out)  # 14

        out = self.conv5_1(out)  # 12
        out = self.norm4(out)
        out = F.relu(out)
        out = self.conv5_2(out)  # 12

        out = F.relu(out)
        out = self.conv5_3(out)  # 12
        out = self.norm4(out)
        out = F.relu(out)
        out = self.maxpool5(out)  # 7

        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)

        return out


class StarDataset(torch.utils.data.Dataset):
    """Return star image and labels"""

    def __init__(self, data_size=50000):
        self.data_size = data_size

    def __len__(self) -> int:
        return self.data_size

    def __getitem__(self, idx) -> t.Tuple[torch.Tensor, torch.Tensor]:
        image, label = synthesize_data(has_star=True)
        return image[None], label

'''
    Customize the loss function
    for the yaw loss first I tranfer the radian to 0 to 360 degree, second use min(abs((output-target))%180,abs(180-abs((output-target))%180) 
    to remove the effect Since rotation has periodicity and mirror pair is shuold be have zero loss such as 0 degree and 180 degree is the same 
    picture.
'''
def my_loss(output, target):
    loss_fn = torch.nn.MSELoss()
    #loss = torch.mean(torch.abs(output[:, 0:2]-target[:, 0:2])+ torch.abs(output[:, 3:5]-target[:, 3:5]))+torch.mean(torch.min(torch.abs(output[:,2]-target[:, 2])%180,torch.abs(180-torch.abs(output[:,2]-target[:, 2])%180)))
    loss = loss_fn (output[:, 0:2],target[:, 0:2])+ loss_fn (output[:, 3:5],target[:, 3:5])+torch.mean(torch.min(torch.abs(output[:,2]-target[:, 2])%180,torch.abs(180-torch.abs(output[:,2]-target[:, 2])%180)))
    yaw_loss = torch.mean(torch.min(torch.abs(output[:,2]-target[:, 2]),torch.abs(180-torch.abs(output[:,2]-target[:, 2]))))
    return loss, yaw_loss


def train(model, dl, num_epochs, optimizer) -> StarModel:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    max_iou=0
    for epoch in range(num_epochs):
        print(f"EPOCH: {epoch}")
        losses = []
        yaw_losses = []
        total_losses=[]
        for image, label in tqdm(dl, total=len(dl)):
            image = image.to(DEVICE).float()
            label = label.to(DEVICE).float()
            preds = model(image)
            loss, yaw_loss = my_loss(preds, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().numpy())
            yaw_losses.append(yaw_loss.detach().cpu().numpy())
            #total_losses.append(total_loss)
        loss_mean = np.mean(losses)
        print("loss_mean:",loss_mean)
        scores = []

        for _ in tqdm(range(1024)):
            image, label = synthesize_data()
            with torch.no_grad():
                pred = model(torch.Tensor(image[None, None]).to(DEVICE))
            np_pred = pred[0].detach().cpu().numpy()
            scores.append(score_iou(np_pred, label))

        ious = np.asarray(scores, dtype="float")
        ious = ious[~np.isnan(ious)]  # remove true negatives
        # only save the model file that have maximum iou value
        if ((ious > 0.7).mean()>max_iou):
            max_iou=(ious > 0.7).mean()
            torch.save(model.state_dict(), "model2.pickle")
        print((ious > 0.7).mean())

    print("the max ious is",max_iou)
    torch.save(model.state_dict(), "model3.pickle")
    return model


def main():
    model = StarModel().to(DEVICE)
    model = nn.DataParallel( model)
    optimizer = torch.optim.Adam(model.parameters())
    star_model = train(
        model,
        torch.utils.data.DataLoader(StarDataset(), batch_size=32, num_workers=2),
        30,
        optimizer,

    )



if __name__ == "__main__":
    main()
