from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter


class E_common(nn.Module):
    def __init__(self, sep, size, dim=512):
        super(E_common, self).__init__()
        self.sep = sep
        self.size = size
        self.dim = dim
        self.layer1 = []
        self.layer2 = []
        self.layer3 = []
        self.layer4 = []
        self.layer5 = []
        self.layer6 = []
        self.z_dim_size = (self.dim - 2 * self.sep) * self.size * self.size

        self.layer1.append(SpectralNorm(nn.Conv2d(3, 32, 4, 2, 1)))
        self.layer1.append(nn.InstanceNorm2d(32))
        self.layer1.append(nn.LeakyReLU(0.2, inplace=True))
        self.l1 = nn.Sequential(*self.layer1)

        self.layer2.append(SpectralNorm(nn.Conv2d(32, 64, 4, 2, 1)))
        self.layer1.append(nn.InstanceNorm2d(64))
        self.layer2.append(nn.LeakyReLU(0.2, inplace=True))
        self.l2 = nn.Sequential(*self.layer2)

        self.layer3.append(SpectralNorm(nn.Conv2d(64, 128, 4, 2, 1)))
        self.layer3.append(nn.InstanceNorm2d(128))
        self.layer3.append(nn.LeakyReLU(0.2, inplace=True))
        self.l3 = nn.Sequential(*self.layer3)

        self.layer4.append(SpectralNorm(nn.Conv2d(128, 256, 4, 2, 1)))
        self.layer4.append(nn.InstanceNorm2d(256))
        self.layer4.append(nn.LeakyReLU(0.2, inplace=True))
        self.l4 = nn.Sequential(*self.layer4)

        self.layer5.append(SpectralNorm(nn.Conv2d(256, (512 - self.sep), 4, 2, 1)))
        self.layer5.append(nn.InstanceNorm2d(512 - self.sep))
        self.layer5.append(nn.LeakyReLU(0.2, inplace=True))
        self.l5 = nn.Sequential(*self.layer5)

        self.layer6.append(SpectralNorm(nn.Conv2d((512 - self.sep), (512 - 2 * self.sep), 4, 2, 1)))
        self.layer6.append(nn.InstanceNorm2d(512 - 2 * self.sep))
        self.layer6.append(nn.LeakyReLU(0.2, inplace=True))
        self.l6 = nn.Sequential(*self.layer6)

    def forward(self, net):
        out = self.l1(net)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.l5(out)
        out = self.l6(out)
        out = out.view(-1, self.z_dim_size)

        return out


class E_separate_A(nn.Module):
    def __init__(self, sep, size):
        super(E_separate_A, self).__init__()
        self.sep = sep
        self.size = size
        self.layer1 = []
        self.layer2 = []
        self.layer3 = []
        self.layer4 = []
        self.layer5 = []
        self.layer6 = []

        self.layer1.append(SpectralNorm(nn.Conv2d(3, 32, 4, 2, 1)))
        self.layer1.append(nn.InstanceNorm2d(32))
        self.layer1.append(nn.LeakyReLU(0.2, inplace=True))
        self.l1 = nn.Sequential(*self.layer1)

        self.layer2.append(SpectralNorm(nn.Conv2d(32, 64, 4, 2, 1)))
        self.layer2.append(nn.InstanceNorm2d(64))
        self.layer2.append(nn.LeakyReLU(0.2, inplace=True))
        self.l2 = nn.Sequential(*self.layer2)

        self.layer3.append(SpectralNorm(nn.Conv2d(64, 128, 4, 2, 1)))
        self.layer3.append(nn.InstanceNorm2d(128))
        self.layer3.append(nn.LeakyReLU(0.2, inplace=True))
        self.l3 = nn.Sequential(*self.layer3)

        self.layer4.append(SpectralNorm(nn.Conv2d(128, 256, 4, 2, 1)))
        self.layer4.append(nn.InstanceNorm2d(256))
        self.layer4.append(nn.LeakyReLU(0.2, inplace=True))
        self.l4 = nn.Sequential(*self.layer4)

        self.layer5.append(SpectralNorm(nn.Conv2d(256, 512, 4, 2, 1)))
        self.layer5.append(nn.InstanceNorm2d(self.sep))
        self.layer5.append(nn.LeakyReLU(0.2, inplace=True))
        self.l5 = nn.Sequential(*self.layer5)

        self.layer6.append(SpectralNorm(nn.Conv2d(512, self.sep, 4, 2, 1)))
        self.layer6.append(nn.InstanceNorm2d(512))
        self.layer6.append(nn.LeakyReLU(0.2, inplace=True))
        self.l6 = nn.Sequential(*self.layer6)

    def forward(self, net):
        out = self.l1(net)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.l5(out)
        out = self.l6(out)
        out = out.view(-1, self.sep * self.size * self.size)
        return out


class E_separate_B(nn.Module):
    def __init__(self, sep, size):
        super(E_separate_B, self).__init__()
        self.sep = sep
        self.size = size
        self.layer1 = []
        self.layer2 = []
        self.layer3 = []
        self.layer4 = []
        self.layer5 = []
        self.layer6 = []

        self.layer1.append(SpectralNorm(nn.Conv2d(3, 32, 4, 2, 1)))
        self.layer1.append(nn.InstanceNorm2d(32))
        self.layer1.append(nn.LeakyReLU(0.2, inplace=True))
        self.l1 = nn.Sequential(*self.layer1)

        self.layer2.append(SpectralNorm(nn.Conv2d(32, 64, 4, 2, 1)))
        self.layer2.append(nn.InstanceNorm2d(64))
        self.layer2.append(nn.LeakyReLU(0.2, inplace=True))
        self.l2 = nn.Sequential(*self.layer2)

        self.layer3.append(SpectralNorm(nn.Conv2d(64, 128, 4, 2, 1)))
        self.layer3.append(nn.InstanceNorm2d(128))
        self.layer3.append(nn.LeakyReLU(0.2, inplace=True))
        self.l3 = nn.Sequential(*self.layer3)

        self.layer4.append(SpectralNorm(nn.Conv2d(128, 256, 4, 2, 1)))
        self.layer4.append(nn.InstanceNorm2d(256))
        self.layer4.append(nn.LeakyReLU(0.2, inplace=True))
        self.l4 = nn.Sequential(*self.layer4)

        self.layer5.append(SpectralNorm(nn.Conv2d(256, 512, 4, 2, 1)))
        self.layer5.append(nn.InstanceNorm2d(self.sep))
        self.layer5.append(nn.LeakyReLU(0.2, inplace=True))
        self.l5 = nn.Sequential(*self.layer5)

        self.layer6.append(SpectralNorm(nn.Conv2d(512, self.sep, 4, 2, 1)))
        self.layer6.append(nn.InstanceNorm2d(512))
        self.layer6.append(nn.LeakyReLU(0.2, inplace=True))
        self.l6 = nn.Sequential(*self.layer6)

    def forward(self, net):
        out = self.l1(net)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.l5(out)
        out = self.l6(out)
        out = out.view(-1, self.sep * self.size * self.size)
        return out


class Decoder(nn.Module):
    def __init__(self, size, dim=512):
        super(Decoder, self).__init__()
        self.size = size
        self.dim = dim

        self.layer1 = []
        self.layer2 = []
        self.layer3 = []
        self.layer4 = []
        self.layer5 = []
        self.layer6 = []

        self.layer1.append(SpectralNorm(nn.ConvTranspose2d(512, 512, 4, 2, 1)))
        self.layer1.append(nn.InstanceNorm2d(512))
        self.layer1.append(nn.ReLU(inplace=True))
        self.l1 = nn.Sequential(*self.layer1)

        self.layer2.append(SpectralNorm(nn.ConvTranspose2d(512, 256, 4, 2, 1)))
        self.layer2.append(nn.InstanceNorm2d(256))
        self.layer2.append(nn.ReLU(inplace=True))
        self.l2 = nn.Sequential(*self.layer2)

        self.layer3.append(SpectralNorm(nn.ConvTranspose2d(256, 128, 4, 2, 1)))
        self.layer3.append(nn.InstanceNorm2d(128))
        self.layer3.append(nn.ReLU(inplace=True))
        self.l3 = nn.Sequential(*self.layer3)

        self.layer4.append(SpectralNorm(nn.ConvTranspose2d(128, 64, 4, 2, 1)))
        self.layer4.append(nn.InstanceNorm2d(64))
        self.layer4.append(nn.ReLU(inplace=True))
        self.l4 = nn.Sequential(*self.layer4)

        self.layer5.append(SpectralNorm(nn.ConvTranspose2d(64, 32, 4, 2, 1)))
        self.layer5.append(nn.InstanceNorm2d(32))
        self.layer5.append(nn.ReLU(inplace=True))
        self.l5 = nn.Sequential(*self.layer5)

        self.layer6.append(nn.ConvTranspose2d(32, 3, 4, 2, 1))
        self.layer6.append(nn.Tanh())
        self.l6 = nn.Sequential(*self.layer6)

    def forward(self, net):
        net = net.view(-1, self.dim, self.size, self.size)
        out = self.l1(net)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.l5(out)
        out = self.l6(out)
        return out


class Disc(nn.Module):
    def __init__(self, sep, size, dim=512):
        super(Disc, self).__init__()
        self.sep = sep
        self.size = size
        self.dim = dim

        self.classify = nn.Sequential(
            nn.Linear((dim - 2 * self.sep) * self.size * self.size, dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, net):
        # net = net.view(-1, (512 - 2 * self.sep) * self.size * self.size)
        net = net.view(-1, (self.dim - 2 * self.sep) * self.size * self.size)
        net = self.classify(net)
        net = net.view(-1)
        return net


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)