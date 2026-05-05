import torch
import torch.nn as nn

# encoder block
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.block(x)
        return self.pool(skip), skip

# decoder block
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)

# full change detection unet
class ChangeDetectionUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        cd  = config["change_detection"]
        in_ch   = cd["in_channels"]    # 6 (t1 + t2 concatenated)
        base_ch = cd["base_channels"]  # 64

        # encoder
        self.enc1 = EncoderBlock(in_ch,      base_ch)
        self.enc2 = EncoderBlock(base_ch,    base_ch * 2)
        self.enc3 = EncoderBlock(base_ch * 2, base_ch * 4)
        self.enc4 = EncoderBlock(base_ch * 4, base_ch * 8)

        # bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch * 8, base_ch * 16, 3, padding=1),
            nn.BatchNorm2d(base_ch * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 16, base_ch * 16, 3, padding=1),
            nn.BatchNorm2d(base_ch * 16),
            nn.ReLU(inplace=True)
        )

        # decoder
        self.dec4 = DecoderBlock(base_ch * 16, base_ch * 8)
        self.dec3 = DecoderBlock(base_ch * 8,  base_ch * 4)
        self.dec2 = DecoderBlock(base_ch * 4,  base_ch * 2)
        self.dec1 = DecoderBlock(base_ch * 2,  base_ch)

        # output head
        self.out_conv = nn.Conv2d(base_ch, 1, 1)

    def forward(self, img_a, img_b):
        # concatenate t1 and t2
        x = torch.cat([img_a, img_b], dim=1)

        # encoder
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)

        # bottleneck
        x = self.bottleneck(x)

        # decoder
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        return self.out_conv(x)
