import torch
import torch.nn as nn
import torch.nn.functional as F

from resnext.resnext101_regular import ResNeXt101

############################################ Initialization ##############################################
def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)) or isinstance(m, nn.GroupNorm):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU) or isinstance(m, nn.MaxPool2d) or isinstance(m, nn.Softmax) or isinstance(m, nn.Sigmoid) or isinstance(m, nn.AdaptiveAvgPool2d) or isinstance(m, nn.ReLU6):
            pass
        elif isinstance(m,nn. ModuleList):
            weight_init(m)
        else:
            m.initialize()

############################################ Basic ##############################################
class basicConv(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(basicConv, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)
    
    def initialize(self):
        weight_init(self)

def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=bias)

# Feature Fusion Module
class FFM(nn.Module):
    def __init__(self, channel):
        super(FFM, self).__init__()
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)

    def forward(self, x_1, x_2):
        out = x_1 * x_2
        #out = torch.cat((x_1, x_2), dim=1)
        out = F.relu(self.bn_1(self.conv_1(out)), inplace=True)
        out = F.relu(self.bn_2(self.conv_2(out)), inplace=True)
        return out

    def initialize(self):
        weight_init(self)

# Cross Aggregation Module
class CAM(nn.Module):
    def __init__(self, channel):
        super(CAM, self).__init__()
        self.down = nn.Sequential(
            conv3x3(channel, channel, stride=2),
            nn.BatchNorm2d(channel)
        )
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)
        self.mul = FFM(channel)

    def forward(self, x_high, x_low):
        left_1 = x_low
        left_2 = F.relu(self.down(x_low), inplace=True)
        right_1 = F.interpolate(x_high, size=x_low.size()[2:], mode='bilinear', align_corners=True)
        right_2 = x_high
        left = F.relu(self.bn_1(self.conv_1(left_1 * right_1)), inplace=True)
        right = F.relu(self.bn_2(self.conv_2(left_2 * right_2)), inplace=True)
        right = F.interpolate(right, size=x_low.size()[2:], mode='bilinear', align_corners=True)
        out = self.mul(left, right)
        return out

    def initialize(self):
        weight_init(self)


####################################### reflection semantic logical module (RSL) ##########################################
# Revised from: PraNet: Parallel Reverse Attention Network for Polyp Segmentation, MICCAI20
# https://github.com/DengPingFan/PraNet
class RFB_modified(nn.Module):
    '''reflection semantic logical module (RSL)'''
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            basicConv(in_channel, out_channel, 1, relu=False),
        )
        self.branch1 = nn.Sequential(
            basicConv(in_channel, out_channel, 1),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, 3, p=7, d=7, relu=False)
        )
        self.branch2 = nn.Sequential(
            basicConv(in_channel, out_channel, 1),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, 3, p=7, d=7, relu=False)
        )

        self.conv_cat = basicConv(3*out_channel, out_channel, 3, p=1, relu=False)
        self.conv_res = basicConv(in_channel, out_channel, 1, relu=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        #x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x
        
    def initialize(self):
        weight_init(self)

########################################### multi-orientation intensity-based contrasted module #########################################
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

    def initialize(self):
        weight_init(self)

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

    def initialize(self):
        weight_init(self)

class CoordAtt(nn.Module):
    # Revised from: Coordinate Attention for Efficient Mobile Network Design, CVPR21
    # https://github.com/houqb/CoordAttention
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
    
    def initialize(self):
        weight_init(self)
        
class SAM(nn.Module):
    def __init__(self, nin: int, nout: int, num_splits: int) -> None:
        super(SAM, self).__init__()

        assert nin % num_splits == 0

        self.nin = nin
        self.nout = nout
        self.num_splits = num_splits

        self.subspaces = nn.ModuleList(
            [CoordAtt(int(self.nin / self.num_splits),int(self.nin / self.num_splits)) for i in range(self.num_splits)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        group_size = int(self.nin / self.num_splits)

        # split at batch dimension
        sub_feat = torch.chunk(x, self.num_splits, dim=1)

        out = []
        for idx, l in enumerate(self.subspaces):
            out.append(self.subspaces[idx](sub_feat[idx]))

        out = torch.cat(out, dim=1)

        return out

    def initialize(self):
        weight_init(self)


class IntensityPositionModule(nn.Module):
    '''multi-orientation intensity-based contrasted module (MIC)'''
    def __init__(self, inplanes, outplanes, g=1):
        super(IntensityPositionModule, self).__init__()
        self.SA1 = SAM(inplanes, outplanes, g)
        self.SA2 = SAM(inplanes, outplanes, g)

        self.conv = nn.Sequential(
                basicConv(inplanes, inplanes, k=3, s=1, p=1, d=1, g=inplanes),
                basicConv(inplanes, outplanes, k=1, s=1, p=0, relu = True)
                )

    def forward(self, x):
        y = x.clone()
        #z = x.clone()
        #u = x.clone()

        y = torch.rot90(y, 1, dims=[2,3])
        y = self.SA1(y)
        y = torch.rot90(y, -1, dims=[2,3])

        #z = torch.rot90(z, 1, dims=[2,3])
        #z = torch.rot90(z, 1, dims=[2,3])
        #z = self.SA3(z)
        #z = torch.rot90(z, -1, dims=[2,3])
        #z = torch.rot90(z, -1, dims=[2,3])

        #u = torch.rot90(u, -1, dims=[2,3])
        #u = self.SA4(u)
        #u = torch.rot90(u, 1, dims=[2,3])

        x = self.SA2(x)

        out = x * y
        out = self.conv(out)

        return out
        
    def initialize(self):
        weight_init(self)


############################################## pooling #############################################
class PyramidPooling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PyramidPooling, self).__init__()
        hidden_channel = int(in_channel / 4)
        self.conv1 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv2 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv3 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv4 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.out = basicConv(in_channel*2, out_channel, k=1, s=1, p=0)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = F.interpolate(self.conv1(F.adaptive_avg_pool2d(x, 1)), size)
        feat2 = F.interpolate(self.conv2(F.adaptive_avg_pool2d(x, 2)), size)
        feat3 = F.interpolate(self.conv3(F.adaptive_avg_pool2d(x, 3)), size)
        feat4 = F.interpolate(self.conv4(F.adaptive_avg_pool2d(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)

        return x
        
    def initialize(self):
        weight_init(self)

##################################### net ###################################################
class Net(nn.Module):
    def __init__(self, cfg, backbone_path="./resnext/resnext_101_32x4d.pth"):
        super(Net, self).__init__()
        self.cfg = cfg
        self.bkbone   = ResNeXt101(backbone_path)

        self.pyramid_pooling = PyramidPooling(2048, 64)

        self.conv1 = nn.ModuleList([
                basicConv(64, 64, k=1, s=1, p=0),
                basicConv(256, 64, k=1, s=1, p=0),
                basicConv(512, 64, k=1, s=1, p=0),
                basicConv(1024, 64, k=1, s=1, p=0),
                basicConv(2048, 64, k=1, s=1, p=0),
                basicConv(2048, 64, k=1, s=1, p=0)
                ])
        self.head = nn.ModuleList([
                conv3x3(64, 1, bias=True), #edge0
                conv3x3(64, 1, bias=True),
                conv3x3(64, 1, bias=True),
                conv3x3(64, 1, bias=True),
                conv3x3(64, 1, bias=True),
                conv3x3(64, 1, bias=True),
        ])
        self.ffm = nn.ModuleList([
                FFM(64),
                FFM(64),
                FFM(64),
                FFM(64),
                FFM(64)
        ])

        self.ipm = nn.ModuleList([
               IntensityPositionModule(64, 64),
               IntensityPositionModule(64, 64),
               IntensityPositionModule(64, 64)
        ])

        self.cam = CAM(64)

        self.ca1 = RFB_modified(1024, 64)
        self.ca2 = RFB_modified(2048, 64)

        self.refine = basicConv(64, 64, k=1, s=1, p=0)
        
        self.initialize()

    def forward(self, x, shape=None): 
        shape = x.size()[2:] if shape is None else shape
        bk_stage1, bk_stage2, bk_stage3, bk_stage4, bk_stage5 = self.bkbone(x)
        
        fused4 = self.pyramid_pooling(bk_stage5)

        f5 = self.ca2(bk_stage5)
        fused4 = F.interpolate(fused4, size=f5.size()[2:], mode='bilinear', align_corners=True)
        fused3 = self.ffm[4](f5, fused4)

        f4 = self.ca1(bk_stage4)
        fused3 = F.interpolate(fused3, size=f4.size()[2:], mode='bilinear', align_corners=True)
        fused2 = self.ffm[3](f4, fused3)

        f3 = self.conv1[2](bk_stage3)
        f3 = self.ipm[2](f3)

        f2 = self.conv1[1](bk_stage2)
        f2 = self.ipm[1](f2)
        f3 = F.interpolate(f3, size=f2.size()[2:], mode='bilinear', align_corners=True)
        fused1 = self.ffm[2](f2, f3)

        fused2 = F.interpolate(fused2, size=[fused1.size(2)//2, fused1.size(3)//2], mode='bilinear', align_corners=True)

        fused1 = self.cam(fused2, fused1)

        f1 = self.conv1[0](bk_stage1)
        f1 = self.ipm[0](f1)
        f2 = F.interpolate(f2, size=f1.size()[2:], mode='bilinear', align_corners=True)
        fused0 = self.ffm[1](f2, f1)

        fused1 = F.interpolate(fused1, size=fused0.size()[2:], mode='bilinear', align_corners=True)
        out0 = self.ffm[0](fused1, fused0)

        out0 = self.refine(out0)
        
        edge0 = F.interpolate(self.head[0](fused0), size=shape, mode='bilinear', align_corners=True)
        out0 = F.interpolate(self.head[1](out0), size=shape, mode='bilinear', align_corners=True)

        if self.cfg.mode == 'train':
            out1 = F.interpolate(self.head[2](fused1), size=shape, mode='bilinear', align_corners=True)
            out2 = F.interpolate(self.head[3](fused2), size=shape, mode='bilinear', align_corners=True)
            out3 = F.interpolate(self.head[4](fused3), size=shape, mode='bilinear', align_corners=True)
            out4 = F.interpolate(self.head[5](fused4), size=shape, mode='bilinear', align_corners=True)
            return out0, edge0, out1, out2, out3, out4
        else:
            return out0, edge0

    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)