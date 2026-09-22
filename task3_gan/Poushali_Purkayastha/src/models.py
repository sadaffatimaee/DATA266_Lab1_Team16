import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        layers = [nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim), nn.ReLU(True)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, ngf=64, n_blocks=6, dropout=0.0):
        super().__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(3, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        dim = ngf
        for _ in range(2):
            layers += [nn.Conv2d(dim, dim * 2, 3, stride=2, padding=1), nn.InstanceNorm2d(dim * 2), nn.ReLU(True)]
            dim *= 2
        layers += [ResidualBlock(dim, dropout) for _ in range(n_blocks)]
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(dim, dim // 2, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(dim // 2),
                nn.ReLU(True),
            ]
            dim //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, 7), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, ndf=64, n_layers=3):
        super().__init__()
        layers = [nn.Conv2d(3, ndf, 4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        dim = ndf
        for _ in range(1, n_layers):
            nxt = min(dim * 2, 512)
            layers += [nn.Conv2d(dim, nxt, 4, stride=2, padding=1), nn.InstanceNorm2d(nxt), nn.LeakyReLU(0.2, True)]
            dim = nxt
        nxt = min(dim * 2, 512)
        layers += [nn.Conv2d(dim, nxt, 4, stride=1, padding=1), nn.InstanceNorm2d(nxt), nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(nxt, 1, 4, stride=1, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def init_weights(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def build_generator(mcfg):
    return ResnetGenerator(int(mcfg["ngf"]), int(mcfg["n_res_blocks"]), float(mcfg["dropout"]))


def build_models(mcfg):
    g_ab = build_generator(mcfg)
    g_ba = build_generator(mcfg)
    d_a = PatchDiscriminator(int(mcfg["ndf"]), int(mcfg["n_d_layers"]))
    d_b = PatchDiscriminator(int(mcfg["ndf"]), int(mcfg["n_d_layers"]))
    for m in (g_ab, g_ba, d_a, d_b):
        m.apply(init_weights)
    return g_ab, g_ba, d_a, d_b


def count_params(module):
    return sum(p.numel() for p in module.parameters())


class ImagePool:
    def __init__(self, size, rng):
        self.size = int(size)
        self.rng = rng
        self.images = []

    def query(self, images):
        if self.size == 0:
            return images
        out = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.images) < self.size:
                self.images.append(img)
                out.append(img)
            elif self.rng.random() > 0.5:
                i = int(self.rng.integers(0, self.size))
                old = self.images[i].clone()
                self.images[i] = img
                out.append(old)
            else:
                out.append(img)
        return torch.cat(out, dim=0)
