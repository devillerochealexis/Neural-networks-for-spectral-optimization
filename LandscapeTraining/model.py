"""
LandscapeModel: neural surrogate that predicts the first
K Dirichlet-Laplacian eigenvalues and their eigenfunctions jointly
Usage
    import torch
    from model import LandscapeModel
    from utils import rayleigh_ratios

    model = LandscapeModel(in_channels=3, n_eig=10, base_ch=32)
    ckpt = torch.load("landscapemodel.pt", map_location="cpu") #all the test were done with a macbook max m1 so can use cpu/mps ; cuda should also be fine
    model.load_state_dict(ckpt["ema_state"])
    model.eval()

    psi_on, log_lam1_umax = model(x, log_umax, mask)
    ratios = rayleigh_ratios(psi_on, mask)                  
    lam1   = torch.exp(log_lam1_umax.squeeze(1)) / umax      
    lam_k  = lam1.unsqueeze(1) * ratios                      
"""

import torch
import torch.nn as nn

from utils import gram_schmidt_omega



class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=False),
            nn.GroupNorm(min(8, out_c), out_c),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(in_c, out_c),
            ConvBNReLU(out_c, out_c),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        h = self.conv(x)
        return self.pool(h), h   


class UpBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.conv = nn.Sequential(
            ConvBNReLU(out_c + skip_c, out_c),
            ConvBNReLU(out_c, out_c),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class LandscapeModel(nn.Module):
    """
    U-Net encoder  decoder, K raw eigenfunction candidates on a 64x64
    grid, followed by Gram-Schmidt orthonormalization in the forward dunction.
    """

    def __init__(self, in_channels=3, n_eig=10, base_ch=32):
        super().__init__()
        c = base_ch
        self.enc1 = DownBlock(in_channels + 1, c)
        self.enc2 = DownBlock(c, c * 2)             
        self.enc3 = DownBlock(c * 2, c * 4)         
        self.enc4 = DownBlock(c * 4, c * 8)         

        self.bottleneck = nn.Sequential(
            ConvBNReLU(c * 8, c * 16),
            ConvBNReLU(c * 16, c * 16),
        )

        # Decoder
        self.dec4 = UpBlock(c * 16, c * 8, c * 8)
        self.dec3 = UpBlock(c * 8, c * 4, c * 4)
        self.dec2 = UpBlock(c * 4, c * 2, c * 2)
        self.dec1 = UpBlock(c * 2, c, c)

        # Eigenfunction head
        self.psi_head = nn.Conv2d(c, n_eig, 1)

        self.lam1_pool = nn.AdaptiveAvgPool2d(1)
        self.lam1_head = nn.Sequential(
            nn.Linear(c * 16 + 1, 64),   # +1 for log_umax
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self.n_eig = n_eig
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, log_umax, mask):
        """
        Returns:
          psi_on
          log_lam1_umax
        """
        indicator, landscape, grad2 = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        eps = 1e-6
        dist_raw = landscape / (torch.sqrt(grad2.clamp(min=0.0)) + eps) * mask
        dist_max = dist_raw.amax(dim=(-2, -1), keepdim=True).clamp(min=eps)
        dist_ch = dist_raw / dist_max
        x4 = torch.cat([x, dist_ch], dim=1)   # (B, in_channels + 1, H, W)

        # Encoder
        x1, s1 = self.enc1(x4)
        x2, s2 = self.enc2(x1)
        x3, s3 = self.enc3(x2)
        x4, s4 = self.enc4(x3)

        # Bottleneck
        b = self.bottleneck(x4)   # (B, 16c, 4, 4)

        # Decoder
        d = self.dec4(b, s4)
        d = self.dec3(d, s3)
        d = self.dec2(d, s2)
        d = self.dec1(d, s1)     

        psi_raw = self.psi_head(d)                      
        psi_on = gram_schmidt_omega(psi_raw, mask)

        # lambda_1 
        b_flat = self.lam1_pool(b).flatten(1)           
        lam1_in = torch.cat([b_flat, log_umax], dim=1)
        log_lam1_umax = self.lam1_head(lam1_in) 

        return psi_on, log_lam1_umax


if __name__ == "__main__":
    # just a naive test
    from utils import rayleigh_ratios

    torch.manual_seed(0)
    B, C, G, K = 2, 3, 64, 10

    model = LandscapeModel(in_channels=C, n_eig=K, base_ch=32)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"LandscapeModel: {n_params:,} parameters")

    x = torch.randn(B, C, G, G)
    mask = (torch.rand(B, 1, G, G) > 0.3).float()
    log_umax = torch.randn(B, 1)

    psi_on, log_lam1_umax = model(x, log_umax, mask)
    ratios = rayleigh_ratios(psi_on, mask)

    print(f"psi_on shape         : {tuple(psi_on.shape)}")

