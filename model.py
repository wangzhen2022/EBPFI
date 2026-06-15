import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import cv2
class DeepLabv3Plus(nn.Module):
    def __init__(self, num_classes=64, backbone='resnet50'):
        super(DeepLabv3Plus, self).__init__()
        
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            self.backbone_channels = 2048
        else:
            raise NotImplementedError

        self.aspp = ASPP(self.backbone_channels, 256)

        self.decoder = Decoder(256, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(64*32*64,64*64)
        )

    def forward(self, x):
        # Extract features
        low_level_features, x = self.backbone_conv(x)
        
        # Apply ASPP
        x = self.aspp(x)
        
        # Apply Decoder
        x = self.decoder(x, low_level_features)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        y = x.view(x.size(0), 64, 64)
        x = F.adaptive_avg_pool2d(y, (1, 64))

        x = torch.flatten(x, 1)
        # exit()
        return x, y
    
    def backbone_conv(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        low_level_features = self.backbone.layer1(x)
        x = self.backbone.layer2(low_level_features)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return low_level_features, x


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)
        self.conv_out = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x1 = self.bn1(self.conv1(x))
        x2 = self.bn2(self.conv2(x))
        x3 = self.bn3(self.conv3(x))
        x4 = self.bn4(self.conv4(x))
        x5 = self.global_avg_pool(x)
        x5 = self.bn5(self.conv5(x5))
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.bn_out(self.conv_out(x))
        
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel_size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class BoundaryGuidedFilter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BoundaryGuidedFilter, self).__init__()
        self.mean_filter = MeanFilter(out_channels)
        self.local_linear_model = LocalLinearModel(out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.cbam = CBAM(out_channels)  # 注意力机制

    def forward(self, W):
        mean_filtered = self.mean_filter(W)
        local_linear = self.local_linear_model(W)
        combined = mean_filtered + local_linear
        # combined = self.bn(combined)  # 融合后添加BatchNorm
        combined = self.cbam(combined)  # 添加CBAM注意力机制
        upsampled = F.interpolate(combined, scale_factor=4, mode='bilinear', align_corners=False)
        return upsampled

class MeanFilter(nn.Module):
    def __init__(self, channels):
        super(MeanFilter, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        return self.avg_pool(x)

class LocalLinearModel(nn.Module):
    def __init__(self, channels):
        super(LocalLinearModel, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.final_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.final_conv(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 48, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 32, kernel_size=1)
        self.bgf = BoundaryGuidedFilter(256, 32)  # num_classes

    def forward(self, x, low_level_features):
        low_level_features = self.conv1(low_level_features)
        low_level_features = self.bn1(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = F.interpolate(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, low_level_features), dim=1)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu(x)
        x = self.conv4(x)

        # 应用Boundary Guided Filter
        x = self.bgf(x)

        return x
    
# class SEBlock(nn.Module):
#     def __init__(self, channels, reduction=16):
#         super(SEBlock, self).__init__()
#         self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         batch, channels, _, _ = x.size()
#         y = F.adaptive_avg_pool2d(x, 1)
#         y = self.fc1(y)
#         y = self.relu(y)
#         y = self.fc2(y)
#         y = self.sigmoid(y)
#         return x * y

# class BoundaryGuidedFilter(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(BoundaryGuidedFilter, self).__init__()
#         self.mean_filter = MeanFilter(out_channels)
#         self.local_linear_model = LocalLinearModel(out_channels)
#         self.se = SEBlock(out_channels)  # 注意力机制

#     def forward(self, W):
#         mean_filtered = self.mean_filter(W)
#         local_linear = self.local_linear_model(W)
#         combined = mean_filtered + local_linear

#         combined = self.se(combined)  # 添加注意力机制
#         upsampled = F.interpolate(combined, scale_factor=4, mode='bilinear', align_corners=False)
#         return upsampled

# class MeanFilter(nn.Module):
#     def __init__(self, channels):
#         super(MeanFilter, self).__init__()
#         self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
    
#     def forward(self, x):
#         return self.avg_pool(x)

# class LocalLinearModel(nn.Module):
#     def __init__(self, channels):
#         super(LocalLinearModel, self).__init__()
#         self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
#         self.bn = nn.BatchNorm2d(channels)
#         self.relu = nn.ReLU()
#         self.final_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
    
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         x = self.final_conv(x)
#         return x

# class Decoder(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super(Decoder, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, 48, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(48)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(256)
#         self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.conv4 = nn.Conv2d(256, 32, kernel_size=1)
#         self.bgf = BoundaryGuidedFilter(256, 32)  # num_classes

#     def forward(self, x, low_level_features):
#         low_level_features = self.conv1(low_level_features)
#         low_level_features = self.bn1(low_level_features)
#         low_level_features = self.relu(low_level_features)

#         x = F.interpolate(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=False)
#         x = torch.cat((x, low_level_features), dim=1)

#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.relu(x)
#         x = self.conv4(x)

#         # 应用Boundary Guided Filter
#         x = self.bgf(x)

#         return x