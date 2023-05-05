import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

from scipy import integrate
import numpy as np
import joblib
from urllib.request import urlretrieve
from numbers import Real
import math
import tqdm

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.distributions.beta import Beta
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from ddsm_lib import *

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = self.main(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        _, _, H, W = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class ScoreNet(nn.Module):
    def __init__(self, ch, ch_mult, attn, num_res_blocks, dropout, time_dependent_weights, time_step=0.01,
                 max_time=4):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = nn.Sequential(GaussianFourierProjection(embed_dim=tdim),
                                            nn.Linear(tdim, tdim))

        self.head = nn.Conv2d(2, ch, kernel_size=3, stride=1, padding=3)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 2, 3, stride=1, padding=1)
        )
        self.initialize()
        self.register_buffer('time_dependent_weights', time_dependent_weights)
        self.time_step = time_step
        self.max_time = max_time

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t):
        x = x.permute(0, 3, 1, 2)
        # Timestep embedding
        temb = self.time_embedding(t / self.max_time)
        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)
        h = h.permute(0, 2, 3, 1)[:, 2:-2, 2:-2,:]
        assert len(hs) == 0
        h = h - h.mean(axis=-1, keepdims=True)
        t_step = (t / self.time_step) - 1
        w0 = self.time_dependent_weights[t_step.long()]
        w1 = self.time_dependent_weights[
            torch.clip(t_step + 1, max=len(self.time_dependent_weights) - 1).long()]
        h = h * (w0 + (t_step - t_step.floor()) * (w1 - w0))[:, None, None, None]

        return h


def load_mnist_binarized(root):
    datapath = os.path.join(root, 'bin-mnist')
    if not os.path.exists(datapath):
        os.makedirs(datapath)
    dataset = os.path.join(datapath, "mnist.pkl.gz")

    if not os.path.isfile(dataset):

        datafiles = {
            "train": "http://www.cs.toronto.edu/~larocheh/public/"
                     "datasets/binarized_mnist/binarized_mnist_train.amat",
            "valid": "http://www.cs.toronto.edu/~larocheh/public/datasets/"
                     "binarized_mnist/binarized_mnist_valid.amat",
            "test": "http://www.cs.toronto.edu/~larocheh/public/datasets/"
                    "binarized_mnist/binarized_mnist_test.amat"
        }
        datasplits = {}
        for split in datafiles.keys():
            print("Downloading %s data..." % (split))
            datasplits[split] = np.loadtxt(urlretrieve(datafiles[split])[0])

        joblib.dump([datasplits['train'], datasplits['valid'], datasplits['test']], open(dataset, "wb"))

    x_train, x_valid, x_test = joblib.load(open(dataset, "rb"))
    return x_train, x_valid, x_test


class BinMNIST(Dataset):
    """Binary MNIST dataset"""

    def __init__(self, data, device='cpu', transform=None):
        h, w, c = 28, 28, 1
        self.device = device
        self.data = torch.tensor(data, dtype=torch.float).view(-1, c, h, w)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample.to(self.device)


def get_binmnist_datasets(root, device='cpu'):
    x_train, x_valid, x_test = load_mnist_binarized(root)
    x_train = np.append(x_train, x_valid, axis=0)
    return BinMNIST(x_train, device=device), BinMNIST(x_valid, device=device), BinMNIST(x_test, device=device)


def binary_to_onehot(x):
    xonehot = []
    xonehot.append((x == 1)[..., None])
    xonehot.append((x == 0)[..., None])
    return torch.cat(xonehot, -1)


if __name__ == '__main__':
    v_one, v_zero, v_one_loggrad, v_zero_loggrad = torch.load('presampled_2C_maxt4.4000steps.100000.pth')
    torch.set_default_dtype(torch.float32)
    alpha = torch.FloatTensor([1.0])
    beta = torch.FloatTensor([1.0])
    timepoints = torch.FloatTensor(np.linspace(0, 4, 4001)[1:]).cuda()

    score_model = ScoreNet(
        ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1, time_step=0.005, time_dependent_weights=torch.load('train_mnist_unet.a2a1bR4.w.pth'))

    score_model.load_state_dict(torch.load('train_mnist_unet.a2a1bR4.best.pth.bak2'))
    score_model = nn.DataParallel(score_model)
    score_model.cuda()
    score_model.eval()

    all_bpds = 0.
    all_items = 0
    min_time = 0.001
    max_time = 4
    batch_size = 512
    sb = UnitStickBreakingTransform()
    device = 'cuda'

    logliks = []
    loglikxs = []
    qlogs = []

    train_set, valid_set, test_set = get_binmnist_datasets('./mnist')
    for _ in range(50):
        tqdm_data = tqdm.tqdm(DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4))
        for x in tqdm_data:
            x = binary_to_onehot(x).squeeze()
            # perturbed_x = sb(Jacobi_Euler_Maruyama_sampler(x[...,0].cpu()*1.0,alpha,beta,min_time,1000,device='cpu')[...,None])
            perturbed_x, _ = diffusion_factory(x.cpu(), torch.zeros(x.shape[0]).long(), v_one, v_zero,
                                               v_one_loggrad, v_zero_loggrad, alpha, beta)
            # perturbed_x, _ = diffusion_fast_flatdirichlet(x.cpu(), min_time, v_one, v_one_loggrad)

            perturbed_x = perturbed_x.to(device)
            x = x.to(device)
            v = sb._inverse(x, prevent_nan=False)
            perturbed_v = sb._inverse(perturbed_x, prevent_nan=False)

            z, loglik = ode_likelihood(perturbed_v,
                                       score_model,
                                       eps=1e-8,
                                       min_time=min_time,
                                       max_time=max_time,
                                       device=device,
                                       alpha=None,
                                       beta=None)
            N = np.prod(x.shape[1:-1])
            loglik = loglik / N

            loglikx = (torch.log(perturbed_x) * (x == 1)).sum(-1).mean()

            torch.set_default_dtype(torch.float64)

            with torch.no_grad():
                qlog = jacobi_diffusion_density(v.double(), perturbed_v.double(), min_time, alpha.to(device),
                                                beta.to(device), order=1000).log()
            for i, (a, b) in enumerate(zip(alpha.to(device), beta.to(device))):
                B = Beta(a, b)
                nanmask = torch.isnan(qlog[..., i])
                qlog[..., i][nanmask] = B.log_prob(perturbed_v[..., i][nanmask].double())
            qlog = (qlog.sum(-1) + sb.log_abs_det_jacobian(perturbed_v)).mean().float()
            torch.set_default_dtype(torch.float32)

            elbo = loglik + loglikx - qlog
            logliks.append(loglik.cpu().detach().numpy())
            loglikxs.append(loglikx.cpu().detach().numpy())
            qlogs.append(qlog.cpu().detach().numpy())

            # bpd = -elbo
            bpd = -(elbo.cpu().detach().numpy()) / np.log(2)
            all_bpds += bpd.sum()
            all_items += bpd.shape[0]
            print("Average bits: {:5f}".format(all_bpds / all_items))
            print("Average nats: {:5f}".format(all_bpds / all_items * np.log(2)))
