import torch
import torch.nn as nn
import torch.nn.functional as F

class SegNet(nn.Module):
    """
    SegNet implementation for semantic segmentation
    """
    def __init__(self, num_classes=1, in_channels=3):
        super(SegNet, self).__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        # Encoder
        self.encoder_conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.encoder_bn1 = nn.BatchNorm2d(64)
        self.encoder_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.encoder_bn2 = nn.BatchNorm2d(64)
        
        self.encoder_conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.encoder_bn3 = nn.BatchNorm2d(128)
        self.encoder_conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.encoder_bn4 = nn.BatchNorm2d(128)
        
        self.encoder_conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.encoder_bn5 = nn.BatchNorm2d(256)
        self.encoder_conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.encoder_bn6 = nn.BatchNorm2d(256)
        self.encoder_conv7 = nn.Conv2d(256, 256, 3, padding=1)
        self.encoder_bn7 = nn.BatchNorm2d(256)
        
        self.encoder_conv8 = nn.Conv2d(256, 512, 3, padding=1)
        self.encoder_bn8 = nn.BatchNorm2d(512)
        self.encoder_conv9 = nn.Conv2d(512, 512, 3, padding=1)
        self.encoder_bn9 = nn.BatchNorm2d(512)
        self.encoder_conv10 = nn.Conv2d(512, 512, 3, padding=1)
        self.encoder_bn10 = nn.BatchNorm2d(512)
        
        self.encoder_conv11 = nn.Conv2d(512, 512, 3, padding=1)
        self.encoder_bn11 = nn.BatchNorm2d(512)
        self.encoder_conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.encoder_bn12 = nn.BatchNorm2d(512)
        self.encoder_conv13 = nn.Conv2d(512, 512, 3, padding=1)
        self.encoder_bn13 = nn.BatchNorm2d(512)
        
        # Decoder
        self.decoder_conv1 = nn.Conv2d(512, 512, 3, padding=1)
        self.decoder_bn1 = nn.BatchNorm2d(512)
        self.decoder_conv2 = nn.Conv2d(512, 512, 3, padding=1)
        self.decoder_bn2 = nn.BatchNorm2d(512)
        self.decoder_conv3 = nn.Conv2d(512, 512, 3, padding=1)
        self.decoder_bn3 = nn.BatchNorm2d(512)
        
        self.decoder_conv4 = nn.Conv2d(512, 512, 3, padding=1)
        self.decoder_bn4 = nn.BatchNorm2d(512)
        self.decoder_conv5 = nn.Conv2d(512, 512, 3, padding=1)
        self.decoder_bn5 = nn.BatchNorm2d(512)
        self.decoder_conv6 = nn.Conv2d(512, 256, 3, padding=1)
        self.decoder_bn6 = nn.BatchNorm2d(256)
        
        self.decoder_conv7 = nn.Conv2d(256, 256, 3, padding=1)
        self.decoder_bn7 = nn.BatchNorm2d(256)
        self.decoder_conv8 = nn.Conv2d(256, 256, 3, padding=1)
        self.decoder_bn8 = nn.BatchNorm2d(256)
        self.decoder_conv9 = nn.Conv2d(256, 128, 3, padding=1)
        self.decoder_bn9 = nn.BatchNorm2d(128)
        
        self.decoder_conv10 = nn.Conv2d(128, 128, 3, padding=1)
        self.decoder_bn10 = nn.BatchNorm2d(128)
        self.decoder_conv11 = nn.Conv2d(128, 64, 3, padding=1)
        self.decoder_bn11 = nn.BatchNorm2d(64)
        
        self.decoder_conv12 = nn.Conv2d(64, 64, 3, padding=1)
        self.decoder_bn12 = nn.BatchNorm2d(64)
        self.decoder_conv13 = nn.Conv2d(64, num_classes, 3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Encoder
        x = self.relu(self.encoder_bn1(self.encoder_conv1(x)))
        x = self.relu(self.encoder_bn2(self.encoder_conv2(x)))
        x, indices1 = F.max_pool2d(x, 2, 2, return_indices=True)
        
        x = self.relu(self.encoder_bn3(self.encoder_conv3(x)))
        x = self.relu(self.encoder_bn4(self.encoder_conv4(x)))
        x, indices2 = F.max_pool2d(x, 2, 2, return_indices=True)
        
        x = self.relu(self.encoder_bn5(self.encoder_conv5(x)))
        x = self.relu(self.encoder_bn6(self.encoder_conv6(x)))
        x = self.relu(self.encoder_bn7(self.encoder_conv7(x)))
        x, indices3 = F.max_pool2d(x, 2, 2, return_indices=True)
        
        x = self.relu(self.encoder_bn8(self.encoder_conv8(x)))
        x = self.relu(self.encoder_bn9(self.encoder_conv9(x)))
        x = self.relu(self.encoder_bn10(self.encoder_conv10(x)))
        x, indices4 = F.max_pool2d(x, 2, 2, return_indices=True)
        
        x = self.relu(self.encoder_bn11(self.encoder_conv11(x)))
        x = self.relu(self.encoder_bn12(self.encoder_conv12(x)))
        x = self.relu(self.encoder_bn13(self.encoder_conv13(x)))
        x, indices5 = F.max_pool2d(x, 2, 2, return_indices=True)
        
        # Decoder
        x = F.max_unpool2d(x, indices5, 2, 2)
        x = self.relu(self.decoder_bn1(self.decoder_conv1(x)))
        x = self.relu(self.decoder_bn2(self.decoder_conv2(x)))
        x = self.relu(self.decoder_bn3(self.decoder_conv3(x)))
        
        x = F.max_unpool2d(x, indices4, 2, 2)
        x = self.relu(self.decoder_bn4(self.decoder_conv4(x)))
        x = self.relu(self.decoder_bn5(self.decoder_conv5(x)))
        x = self.relu(self.decoder_bn6(self.decoder_conv6(x)))
        
        x = F.max_unpool2d(x, indices3, 2, 2)
        x = self.relu(self.decoder_bn7(self.decoder_conv7(x)))
        x = self.relu(self.decoder_bn8(self.decoder_conv8(x)))
        x = self.relu(self.decoder_bn9(self.decoder_conv9(x)))
        
        x = F.max_unpool2d(x, indices2, 2, 2)
        x = self.relu(self.decoder_bn10(self.decoder_conv10(x)))
        x = self.relu(self.decoder_bn11(self.decoder_conv11(x)))
        
        x = F.max_unpool2d(x, indices1, 2, 2)
        x = self.relu(self.decoder_bn12(self.decoder_conv12(x)))
        x = self.decoder_conv13(x)
        return x

    def get_gradcam_target_layer(self):
        return self.encoder_conv13

if __name__ == "__main__":
    model = SegNet(num_classes=1, in_channels=3)
    x = torch.randn(1, 3, 512, 512)
    out = model(x)
    print("SegNet output shape:", out.shape)
