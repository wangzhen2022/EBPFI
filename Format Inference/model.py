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

class BoundaryGuidedFilter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BoundaryGuidedFilter, self).__init__()
        self.mean_filter = MeanFilter(out_channels)
        self.local_linear_model = LocalLinearModel(out_channels)

    def forward(self, W):
        mean_filtered = self.mean_filter(W)
        local_linear = self.local_linear_model(W)
        combined = mean_filtered + local_linear
        # print(combined.shape)
        upsampled = F.interpolate(combined, scale_factor=4, mode='bilinear', align_corners=False)
        # print(upsampled.shape)
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
        self.conv = nn.Conv2d(channels,channels , kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.final_conv = nn.Conv2d(channels, channels, kernel_size=1,  bias=False)
    
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
        self.bgf = BoundaryGuidedFilter(256, 32)#num_classes

    def forward(self, x, low_level_features):
        low_level_features = self.conv1(low_level_features)
        low_level_features = self.bn1(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = F.interpolate(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, low_level_features), dim=1)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)

        # Apply Boundary Guided Filter
        x = self.bgf(x)

        return x

# class MeanFilter(nn.Module):
#     def __init__(self, channels):
#         super(MeanFilter, self).__init__()
#         self.channels = channels
        
#     def forward(self, x):
#         # Detach the tensor from the computation graph and convert to numpy
#         n, c, h, w = x.shape
#         if c == 1:
#             x_np = x.squeeze(1).detach().cpu().numpy()  # Shape: (n, h, w)
#         else:
#             x_np = x.permute(0, 2, 3, 1).detach().cpu().numpy()  # Shape: (n, h, w, c)
        
#         blurred_list = []
#         for img in x_np:
#             if self.channels == 1:
#                 img_blurred = cv2.blur(img, (5, 5))
#                 blurred_list.append(torch.tensor(img_blurred).unsqueeze(0))
#             else:
#                 img_blurred = cv2.blur(img, (5, 5))
#                 blurred_list.append(torch.tensor(img_blurred).permute(2, 0, 1))
        
#         blurred = torch.stack(blurred_list).to(x.device)
#         return blurred