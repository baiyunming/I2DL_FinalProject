#define model

import torch.nn as nn
import torch
import torchvision


class YOLO_Resnet(nn.Module):
    def __init__(self, feat_dim=2048, S=5, C=4, B=2):
        super(YOLO_Resnet, self).__init__()

        self.feat_dim = feat_dim
        self.S = S
        self.C = C
        self.B = B

        self.backbone = torchvision.models.resnet50(pretrained=True)

        # # Fix Initial Layers
        for p in list(self.backbone.children())[:-1]:
            p.requires_grad = False

        # # get the structure until the Fully Connected Layer
        modules = list(self.backbone.children())[:-1]
        self.backbone = nn.Sequential(*modules)

        # Add new fully connected layers
        self.fc1 = nn.Linear(feat_dim, 1024)  # 2048 -> 1024
        self.fc2 = nn.Linear(1024, self.S * self.S * (self.C + self.B * 5))  # 1024 -> 5*5*(4+2*5)
        self.dropout = nn.Dropout(p=0.3)

        #self.bn1 = nn.BatchNorm1d(1024)
        self.activation = nn.ReLU()

        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.fc1.bias, 0.1)

        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.fc2.bias, 0.1)

    def forward(self, img):
        batch_size = img.shape[0]
        out = self.backbone(img)  # get the feature from the pre-trained resnet
        out = self.dropout(self.activation(self.fc1(out.view(batch_size, -1))))
        out = self.fc2(out)

        return out




# def test():
#     device = 'cuda'
#     model = YOLO_Resnet().to(device)
#     x = torch.randn((2,3,224,224))
#     x = x.to(device)
#     print(model(x).shape)
#
#
# test()