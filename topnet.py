import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math


tree_arch = {}
tree_arch[2] = [32, 64]
tree_arch[4] = [4, 8, 8, 8]
tree_arch[6] = [2, 4, 4, 4, 4, 4]
tree_arch[8] = [2, 2, 2, 2, 2, 4, 4, 4]


def get_arch(nlevels, npts):
    logmult = int(math.log2(npts/2048))
    assert 2048*(2**(logmult)) == npts, "Number of points is %d, expected 2048x(2^n)" % (npts)
    arch = tree_arch[nlevels]
    while logmult > 0:
        last_min_pos = np.where(arch==np.min(arch))[0][-1]
        arch[last_min_pos]*=2
        logmult -= 1
    return arch


class mlp(nn.Module):
    def __init__(self, in_num, out_num):
        super(mlp, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_num,256,1)
        self.conv2 = torch.nn.Conv1d(256,64,1)
        self.conv3 = torch.nn.Conv1d(64,out_num,1)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


class final_mlp(nn.Module):
    def __init__(self, in_num, out_num):
        super(final_mlp, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_num,256,1)
        self.conv2 = torch.nn.Conv1d(256,64,1)
        self.conv3 = torch.nn.Conv1d(64,out_num,1)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class TopNet_decoder(nn.Module):
    def __init__(self, arch=[4,4,8,8]):
        super(TopNet_decoder, self).__init__()
        self.arch = arch
        self.level_num = len(arch)
        self.level_1 = nn.ModuleList()
        self.level_2 = nn.ModuleList()
        self.level_3 = nn.ModuleList()
        self.level_4 = nn.ModuleList()
        # construct tree
        for _ in range(arch[0]):
            self.level_1.append(mlp(512,8))
        for _ in range(arch[1]):
            self.level_2.append(mlp(512+8,8))
        for _ in range(arch[2]):
            self.level_3.append(mlp(512+8,8))
        for _ in range(arch[3]):
            self.level_4.append(final_mlp(512+8,3))
    def forward(self, x):
        features_1 = []
        for net in self.level_1:
            features_1.append(torch.cat([net(x), x], dim=1))
        # print(features_1[0].shape)
        features_2 = []
        for feat in features_1:
            for net in self.level_2:
                features_2.append(torch.cat([net(feat), x], dim=1))
        # print(len(features_2))
        features_3 = []
        for feat in features_2:
            for net in self.level_3:
                features_3.append(torch.cat([net(feat), x], dim=1))
        # print(len(features_3))
        features_4 = []
        for feat in features_3:
            for net in self.level_4:
                features_4.append(net(feat))
        # print(len(features_4))
        pc = torch.cat(features_4, dim=2)
        # print(pc.shape)
        return pc


class PCN_encoder(nn.Module):
    def __init__(self, input_num, AorB):
        super(PCN_encoder, self).__init__()
        self.i_num = input_num
        self.AorB  = AorB
        self.conv1 = torch.nn.Conv1d(3,128,1) # [in_channels, out_channels, kernel_size]
        self.conv2 = torch.nn.Conv1d(128,256,1)
        self.conv3 = torch.nn.Conv1d(512,512,1)
        self.conv4 = torch.nn.Conv1d(512,512,1)

    def forward(self, x):
        """
         input x is a batch of partial point clouds whose shapes are
          [batch_size, 3(channels), partial_point_cloud_number]
        """
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        global_feature_1 = torch.max(x, dim=2, keepdim=True)[0].repeat(1, 1, self.i_num)
        x = torch.cat([x, global_feature_1], dim=1)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        global_feature_2 = None
        if self.AorB == 'a':
            global_feature_2 = torch.max(x[:,:,0:256], dim=2, keepdim=True)[0] # [batch_size, 1024, 1]
        if self.AorB == 'b':
            global_feature_2 = torch.max(x[:,:,-256:], dim=2, keepdim=True)[0] # [batch_size, 1024, 1]
        return global_feature_2 


class TopNet(nn.Module):
    def __init__(self, input_num, AorB):
        super(TopNet, self).__init__()
        self.encoder = PCN_encoder(input_num, AorB)
        self.decoder = TopNet_decoder()
    def forward(self, x):
        feat = self.encoder(x)
        pc = self.decoder(feat)
        return pc


class TopNet_path1(nn.Module):
    def __init__(self, input_num=512, coarse_num=2048):
        super(TopNet_path1, self).__init__()
        self.topnet1 = TopNet(input_num,'a')
        self.topnet2 = TopNet(int(input_num/2)+int(coarse_num/2),'b')
        self.input_num = input_num
    def forward(self, x):
        A_1 = self.topnet1(x)
        B_2 = self.topnet2(torch.cat([A_1, x[:,:,int(self.input_num/2):]], dim=2))
        return A_1, B_2


class TopNet_path2(nn.Module):
    def __init__(self, input_num=512, coarse_num=2048):
        super(TopNet_path2, self).__init__()
        self.topnet1 = TopNet(input_num,'b')
        self.topnet2 = TopNet(int(input_num/2)+int(coarse_num/2),'a')
        self.input_num = input_num
    def forward(self, x):
        B_1 = self.topnet1(x)
        A_2 = self.topnet2(torch.cat([x[:,:,0:int(self.input_num/2)],B_1], dim=2))
        return B_1, A_2



if __name__ == '__main__':
    # print(get_arch(8, 16384))
    model  = TopNet()
    model.cuda()
    input_x = torch.randn(4,3,256).cuda()
    out = model(input_x)
    print(out.shape)