import torch
import torch
import torch.nn as nn
import config
from torchsummary import summary
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        
        # Initial downsampling block
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),  # 512x512 -> 256x256
            nn.LeakyReLU(0.2),
        )
        
        # Downsample blocks
        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)  # 256x256 -> 128x128
        self.down2 = Block(features * 2, features * 4, down=True, act="leaky", use_dropout=False)  # 128x128 -> 64x64
        self.down3 = Block(features * 4, features * 8, down=True, act="leaky", use_dropout=False)  # 64x64 -> 32x32
        self.down4 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)  # 32x32 -> 16x16
        self.down5 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)  # 16x16 -> 8x8
        self.down6 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)  # 8x8 -> 4x4
        
        # Bottleneck block
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1),  # 4x4 -> 2x2
            nn.ReLU()
        )
        
        # Upsample blocks
        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)  # 2x2 -> 4x4
        self.up2 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)  # 4x4 -> 8x8
        self.up3 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)  # 8x8 -> 16x16
        self.up4 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False)  # 16x16 -> 32x32
        self.up5 = Block(features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False)  # 32x32 -> 64x64
        self.up6 = Block(features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False)  # 64x64 -> 128x128
        self.up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)  # 128x128 -> 256x256
        
        # Final upsampling block
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),  # 256x256 -> 512x512
            nn.Tanh(),
       

        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.final_up(torch.cat([up7, d1], 1))


def test():
    x = torch.randn((1, 3, 512, 512)).to(config.DEVICE)
    # x = x.to(config.DEVICE)
    model = Generator(in_channels=3, features=64).to(config.DEVICE)
    # Load the pre-trained weights
    checkpoint = torch.load(config.CHECKPOINT_GEN, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['state_dict'],strict=False)

    # Print the model summary
    summary(model, input_size=(3, 512, 512))
    preds = model(x)
    print(preds.shape)

if __name__ == "__main__":
    test()