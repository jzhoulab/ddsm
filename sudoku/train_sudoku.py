import numpy as np
import random
import math
from matplotlib import pyplot as plt
import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import sys 
sys.path.append("../")
from ddsm import *

def worker_init_fn(worker_id):
    np.random.seed(worker_id)


#############################################
############ SUDOKU DATASET #################
#############################################
def construct_puzzle_solution():
    # Loop until we're able to fill all 81 cells with numbers, while
    # satisfying the constraints above.
    while True:
        try:
            puzzle = [[0] * 9 for i in range(9)]  # start with blank puzzle
            rows = [set(range(1, 10)) for i in range(9)]  # set of available
            columns = [set(range(1, 10)) for i in range(9)]  # numbers for each
            squares = [set(range(1, 10)) for i in range(9)]  # row, column and square
            for i in range(9):
                for j in range(9):
                    # pick a number for cell (i,j) from the set of remaining available numbers
                    choices = rows[i].intersection(columns[j]).intersection(squares[(i // 3) * 3 + j // 3])
                    choice = random.choice(list(choices))

                    puzzle[i][j] = choice

                    rows[i].discard(choice)
                    columns[j].discard(choice)
                    squares[(i // 3) * 3 + j // 3].discard(choice)

            # success! every cell is filled.
            return puzzle

        except IndexError:
            # if there is an IndexError, we have worked ourselves in a corner (we just start over)
            pass


def gen_sudoku(num):
    """
    Generates `num` games of Sudoku.
    """
    solutions = np.zeros((num, 9, 9), np.int32)
    for i in range(num):
        solutions[i] = construct_puzzle_solution()

    return solutions


class SudokuDataset(Dataset):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __len__(self):
        return int(self.batch_size * 1000)

    def __getitem__(self, idx):
        sudoku = gen_sudoku(1)
        dataset = np.eye(9)[sudoku.reshape(sudoku.shape[0], -1) - 1]
        return dataset


def sudoku_acc(sample, return_array=False):
    sample = sample.detach().cpu().numpy()
    correct = 0
    total = sample.shape[0]
    ans = sample.argmax(-1) + 1
    numbers_1_N = np.arange(1, 9 + 1)
    corrects = []
    for board in ans:
        if (np.all(np.sort(board, axis=1) == numbers_1_N) and
                np.all(np.sort(board.T, axis=1) == numbers_1_N)):
            # Check blocks

            blocks = board.reshape(3, 3, 3, 3).transpose(0, 2, 1, 3).reshape(9, 9)
            if np.all(np.sort(board.T, axis=1) == numbers_1_N):
                correct += 1
                corrects.append(True)
            else:
                corrects.append(False)
        else:
            corrects.append(False)

    if return_array:
        return corrects
    else:
        print('correct {} %'.format(100 * correct / total))

#############################################
############## SUDOKU MDDEL #################
#############################################

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, n_head, bias=None):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

        self.n_head = n_head
        self.n_embd = n_embd
        self.register_buffer("bias", bias)

        self.bias_proj = nn.Linear(bias.shape[-1], n_head)  # T, T, nh

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att + self.bias_proj(self.bias).permute((2, 0, 1))
        att = F.softmax(att, dim=-1)
        # att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, bias=None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, bias=bias)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(n_embd, 4 * n_embd),
            c_proj=nn.Linear(4 * n_embd, n_embd),
            act=NewGELU(),
        ))
        m = self.mlp

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp.c_proj(self.mlp.act(self.mlp.c_fc(self.ln_2(x))))
        return x


class GaussianFourierProjection(nn.Module):
    """
    Gaussian random features for encoding time steps.
    """

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """
    A fully connected layer that reshapes outputs to feature maps.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[...]


class ScoreNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, allenc_relative, embed_dim=256):
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))

        self.linear = Dense(9, 128)
        self.blocks = nn.ModuleList(Block(128, 8, bias=allenc_relative) for _ in range(20))
        self.denses = nn.ModuleList(Dense(embed_dim, 128) for _ in range(20))
        self.act = NewGELU()
        self.softplus = nn.Softplus()
        self.output = Dense(128, 9)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x, t):
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.embed(t))

        # Encoding path
        h = self.linear(x.view(-1, 81, 9))
        for le, ld in zip(self.blocks, self.denses):
            h = le(h + ld(embed)[:, None, :])

        h = self.output(h)

        # h = h.reshape(x.size()) * torch.exp(-t[:,None,None,None]* self.softplus(self.scale)) /  ((x+1e-6)*(1-x+1e-6))
        h = h.reshape(
            x.size())  # * torch.exp(-t[:,None,None,None]* self.softplus(self.scale)) * (1/(x+1e-3)+1/(1-x+1e-3))
        h = h - h.mean(axis=-1, keepdims=True)
        return h


def define_relative_encoding(): 
    colind = np.array([
                    [0,1,2,3,4,5,6,7,8],
                    [0,1,2,3,4,5,6,7,8],
                    [0,1,2,3,4,5,6,7,8],
                    [0,1,2,3,4,5,6,7,8],
                    [0,1,2,3,4,5,6,7,8],
                    [0,1,2,3,4,5,6,7,8],
                    [0,1,2,3,4,5,6,7,8],
                    [0,1,2,3,4,5,6,7,8],
                    [0,1,2,3,4,5,6,7,8]
    ])

    rowind = np.array([
                [0,0,0,0,0,0,0,0,0],
                [1,1,1,1,1,1,1,1,1],
                [2,2,2,2,2,2,2,2,2],
                [3,3,3,3,3,3,3,3,3],
                [4,4,4,4,4,4,4,4,4],
                [5,5,5,5,5,5,5,5,5],
                [6,6,6,6,6,6,6,6,6],
                [7,7,7,7,7,7,7,7,7],
                [8,8,8,8,8,8,8,8,8]
    ])


    blockind = np.array([
                [0,0,0,1,1,1,2,2,2],
                [0,0,0,1,1,1,2,2,2],
                [0,0,0,1,1,1,2,2,2],
                [3,3,3,4,4,4,5,5,5],
                [3,3,3,4,4,4,5,5,5],
                [3,3,3,4,4,4,5,5,5],
                [6,6,6,7,7,7,8,8,8],
                [6,6,6,7,7,7,8,8,8],
                [6,6,6,7,7,7,8,8,8]
    ])

    colenc = np.zeros((81, 9))
    rowenc = np.zeros((81, 9))
    blockenc = np.zeros((81, 9))
    colenc[np.arange(81), colind.flatten()] = 1
    rowenc[np.arange(81), rowind.flatten()] = 1
    blockenc[np.arange(81),blockind.flatten()] = 1
    allenc = np.concatenate([colenc, rowenc, blockenc], axis=1)
    return torch.FloatTensor(allenc[:,None,:] == allenc[None,:,:])

if __name__ == "__main__":
    device = 'cuda'
    batch_size = 256
    num_workers = 16

    lr = 1e-4
    num_steps = 500
    n_epochs = 600
    random_order = False

    v_one, v_zero, v_one_loggrad, v_zero_loggrad, timepoints =  torch.load('steps400.cat9.time1.0.samples100000.pth') 
    n_timesteps = timepoints.shape[0]
    alpha = torch.FloatTensor([1.0])
    beta = torch.FloatTensor([8.0])
    torch.set_default_dtype(torch.float32)


    sb = UnitStickBreakingTransform()

    # Estimate timepoints
    train_dataloader = DataLoader(SudokuDataset(batch_size),
                                  batch_size, shuffle=True,
                                  num_workers=num_workers,
                                  worker_init_fn=worker_init_fn)

    time_dependent_cums = torch.zeros(n_timesteps).to(device)
    time_dependent_counts = torch.zeros(n_timesteps).to(device)

    avg_loss = 0.
    num_items = 0
    for i, x in enumerate(train_dataloader):
        x = x.reshape(-1, 9, 9, 9)
        random_t = torch.randint(0, n_timesteps, (x.shape[0],))
        order = np.random.permutation(np.arange(9))
        perturbed_x, perturbed_x_grad = diffusion_fast_flatdirichlet(x[..., order], random_t, v_one, v_one_loggrad)
        perturbed_x = perturbed_x[..., np.argsort(order)]
        perturbed_x_grad = perturbed_x_grad[..., np.argsort(order)]
        perturbed_x = perturbed_x.to(device)
        perturbed_x_grad = perturbed_x_grad.to(device)
        random_t = random_t.to(device)
        perturbed_v = sb._inverse(perturbed_x)

        order = np.random.permutation(np.arange(9))
        perturbed_v = sb._inverse(perturbed_x[..., order], prevent_nan=True).detach()

        time_dependent_counts[random_t] += 1
        time_dependent_cums[random_t] += (perturbed_v * (1 - perturbed_v) * (
            gx_to_gv(perturbed_x_grad[..., order], perturbed_x[..., order])) ** 2).mean(dim=(1, 2, 3)).detach()

        if i > 20:
            break

    time_dependent_weights = time_dependent_cums / time_dependent_counts
    time_dependent_weights = time_dependent_weights / time_dependent_weights.mean()

    plt.plot(torch.sqrt(time_dependent_weights.cpu()))
    plt.savefig("timedependent_weight.png")

    # Train code
    score_model = ScoreNet(define_relative_encoding())
    score_model = score_model.to('cuda')
    optimizer = Adam(score_model.parameters(), lr=lr)

    train_dataloader = DataLoader(SudokuDataset(batch_size),
                                  batch_size, shuffle=True,
                                  num_workers=num_workers,
                                  worker_init_fn=worker_init_fn)

    tqdm_epoch = tqdm.trange(n_epochs)
    j = 0
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        score_model.train()
        for dataset in train_dataloader:

            x = dataset.reshape(-1, 9, 9, 9)
            random_t = torch.LongTensor(np.random.choice(np.arange(n_timesteps), size=x.shape[0], p=(
                        time_dependent_weights / time_dependent_weights.sum()).cpu().detach().numpy()))

            order = np.random.permutation(np.arange(9))
            if random_order:
                perturbed_x, perturbed_x_grad = diffusion_fast_flatdirichlet(x[..., order], random_t, v_one,
                                                                             v_one_loggrad)
                perturbed_x = perturbed_x[..., np.argsort(order)]
                perturbed_x_grad = perturbed_x_grad[..., np.argsort(order)]
            else:
                perturbed_x, perturbed_x_grad = diffusion_fast_flatdirichlet(x, random_t, v_one, v_one_loggrad)

            perturbed_x = perturbed_x.to(device)
            perturbed_x_grad = perturbed_x_grad.to(device)

            score = score_model(perturbed_x, timepoints[random_t].to(device))
            order = np.random.permutation(np.arange(9))

            if random_order:
                perturbed_v = sb._inverse(perturbed_x[..., order], prevent_nan=True).detach()
                loss = torch.mean(torch.mean(
                    1 / (time_dependent_weights)[random_t, None, None, None] * perturbed_v * (1 - perturbed_v) * (
                                gx_to_gv(score[..., order], perturbed_x[..., order], create_graph=True) - gx_to_gv(
                            perturbed_x_grad[..., order], perturbed_x[..., order])) ** 2, dim=(1, 2)))
            else:
                perturbed_v = sb._inverse(perturbed_x, prevent_nan=True).detach()
                loss = torch.mean(torch.mean(
                    1 / (time_dependent_weights)[random_t, None, None, None] * perturbed_v * (1 - perturbed_v) * (
                                gx_to_gv(score, perturbed_x, create_graph=True) - gx_to_gv(perturbed_x_grad,
                                                                                           perturbed_x)) ** 2,
                    dim=(1, 2)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]

        print('Training loss', avg_loss / num_items)

        sampler = Euler_Maruyama_sampler

        score_model.eval()
        samples = sampler(score_model,
                          (9, 9, 9),
                          batch_size=256,
                          max_time=1,
                          time_dilation=1,
                          num_steps=200,
                          random_order=False,
                          speed_balanced=False,
                          device=device)
        sudoku_acc(samples)
        j += 1
