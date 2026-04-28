import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size=14, train_cnn=False):
        super().__init__()
        self.enc_image_size = encoded_image_size

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        features = list(vgg.features.children())

        # Keep convolutional layers only; stop before final maxpool to preserve spatial map better.
        self.cnn = nn.Sequential(*features[:-1])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        for p in self.cnn.parameters():
            p.requires_grad = train_cnn

    def forward(self, images):
        """
        images: (B, 3, H, W)
        returns: (B, num_pixels, encoder_dim)
        """
        out = self.cnn(images)                      # (B, 512, H', W')
        out = self.adaptive_pool(out)               # (B, 512, 14, 14)
        out = out.permute(0, 2, 3, 1)               # (B, 14, 14, 512)
        out = out.view(out.size(0), -1, out.size(-1))  # (B, 196, 512)
        return out
