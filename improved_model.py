import torch
import torch.nn as nn
import torch.nn.functional as F

# 首先定义残差块
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

# FPN
class FeaturePyramidNetwork(nn.Module):
    def __init__(self, channels=64, num_res_blocks=1):
        super().__init__()
        
        # 自顶向下
        self.down1 = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, stride=1, padding=1),
            ResBlock(channels)
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(channels, channels*2, kernel_size=3, stride=1, padding=1),
            ResBlock(channels*2)
        )
        
        self.down3 = nn.Sequential(
            nn.Conv2d(channels*2, channels*4, kernel_size=3, stride=1, padding=1),
            ResBlock(channels*4)
        )
        
        # 横向
        self.lateral1 = nn.Conv2d(channels*4, channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(channels*2, channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(channels, channels, kernel_size=1)
        
        # 简平滑
        self.smooth1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        c1 = self.down1(x)
        c2 = self.down2(c1)
        c3 = self.down3(c2)
        
        p3 = self.lateral1(c3)
        p3_upsampled = F.interpolate(p3, size=c2.shape[2:], mode='bilinear', align_corners=True)
        p2 = self.lateral2(c2) + p3_upsampled
        
        p2_upsampled = F.interpolate(p2, size=c1.shape[2:], mode='bilinear', align_corners=True)
        p1 = self.lateral3(c1) + p2_upsampled
        
        p2 = self.smooth1(p2)
        p1 = self.smooth2(p1)
        
        return [p1, p2, p3]

# 自注意力
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return torch.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = torch.sigmoid(avg_out + max_out)
        return out
    

class Generator(nn.Module):
    def __init__(self, scale_factor=1.0, channels=64):
        self.scale_factor = scale_factor
        super().__init__()
        self.fpn = FeaturePyramidNetwork(channels=channels)
        self.spatial_attention = SpatialAttention()
        self.channel_attention = ChannelAttention(channels)
        
        # 特征融合
        self.fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                ResBlock(channels)
            ) for _ in range(3)
        ])
        
        # 后处理
        self.post_fusion = nn.Sequential(
            ResBlock(channels * 3),
            nn.Conv2d(channels * 3, channels, 1)
        )
        
        # 最终输出
        self.final = nn.Conv2d(channels, 1, 1)
        
    def forward(self, x):
        if self.scale_factor != 1.0:
            input_size = x.shape[2:]
            x = F.interpolate(x, scale_factor=self.scale_factor, 
                            mode='bilinear', align_corners=True)
        
        features = self.fpn(x)
        
        enhanced_features = []
        for i, feat in enumerate(features):
            spatial_weight = self.spatial_attention(feat)
            channel_weight = self.channel_attention(feat)
            feat = feat * spatial_weight * channel_weight
            
            feat = self.fusion[i](feat)
            
            if i > 0:
                feat = F.interpolate(feat, size=features[0].shape[2:], 
                                   mode='bilinear', align_corners=True)
            
            enhanced_features.append(feat)
        
        final_feature = torch.cat(enhanced_features, dim=1)
        final_feature = self.post_fusion(final_feature)
        output = self.final(final_feature)
        
        if self.scale_factor != 1.0:
            output = F.interpolate(output, size=input_size, 
                                 mode='bilinear', align_corners=True)
        
        return output

class discriminator(nn.Module):
    def __init__(self, mini_batch_size=10):
        self.mini_batch_size = mini_batch_size
        super(discriminator, self).__init__()
        self.leakyrelu1 = nn.LeakyReLU(0.2)
        self.leakyrelu2 = nn.LeakyReLU(0.2)
        self.leakyrelu3 = nn.LeakyReLU(0.2)
        self.leakyrelu4 = nn.LeakyReLU(0.2)
        self.Tanh1 = nn.Tanh()
        self.Tanh2 = nn.Tanh()
        self.Softmax = nn.Softmax()


        self.d_conv1 = nn.Conv2d(2,  24, 3, dilation=1, padding=1)
        self.d_conv2 = nn.Conv2d(24, 24, 3, dilation=1, padding=1)
        self.d_conv3 = nn.Conv2d(24, 24, 3, dilation=1, padding=1)
        self.d_conv4 = nn.Conv2d(24, 1,  3, dilation=1, padding=1)

        self.d_bn1 = nn.BatchNorm2d(24)
        self.d_bn2 = nn.BatchNorm2d(24)
        self.d_bn3 = nn.BatchNorm2d(24)
        self.d_bn4 = nn.BatchNorm2d(1)
        self.d_bn5 = nn.BatchNorm2d(128)
        self.d_bn6 = nn.BatchNorm2d(64)
        self.d_bn7 = nn.BatchNorm2d(3)
        
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)


    def forward(self, input_images): # 输入[3B, 2, 128, 128],输出[B, 1, 128, 128]
        net = F.max_pool2d(input_images, kernel_size=[2, 2])  # [3B, 2, 64, 64]
        net = F.max_pool2d(net, kernel_size=[2, 2])  # [3B, 2, 32, 32]
        
        net = self.d_conv1(net)
        net = self.d_bn1(net)
        net = self.leakyrelu1(net)

        net = self.d_conv2(net)
        net = self.d_bn2(net)
        net = self.leakyrelu2(net)

        net = self.d_conv3(net)
        net = self.d_bn3(net)
        net = self.leakyrelu3(net)

        net = self.d_conv4(net)
        net = self.d_bn4(net)
        net1 = self.leakyrelu4(net) # [3B, 1, 32, 32]

        net = net1.view(-1, 1024) # [3B, 1024]
        net = self.fc1(net)      # [3B, 128]
        net = net.unsqueeze(2).unsqueeze(3)
        net = self.d_bn5(net)
        net = self.Tanh1(net)    # [3B, 128, 1, 1]

        net = net.view(-1, 128) # [3B, 128]
        net = self.fc2(net)      # [3B, 64]
        net = net.unsqueeze(2).unsqueeze(3)
        net = self.d_bn6(net)
        net = self.Tanh2(net)   # [3B, 64, 1, 1]

        net = net.view(-1, 64) # [3B, 64]
        net = self.fc3(net)      # [3B, 3]
        net = net.unsqueeze(2).unsqueeze(3)
        net = self.d_bn7(net)
        net = self.Softmax(net) # [3B, 3, 1, 1]
        net = net.squeeze(3).squeeze(2)
        
        realscore0, realscore1, realscore2 = torch.split(net, self.mini_batch_size, dim=0)
        feat0, feat1, feat2 = torch.split(net1, self.mini_batch_size, dim=0)
        featDist = torch.mean(torch.pow(feat1 - feat2, 2))

        return realscore0, realscore1, realscore2, featDist
    
    