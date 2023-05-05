import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

import time
import tqdm
import tabix
import pyBigWig
import pandas as pd
from matplotlib import pyplot as plt

from selene_sdk.utils import NonStrandSpecific
from selene_sdk.targets import Target

from sei import *
from selene_utils2 import *
from ddsm_lib import *

import logging

logger = logging.getLogger()


class ModelParameters:
    seifeatures_file = '/archive/bioinformatics/Zhou_lab/shared/jzhou/sei-framework/model/target.names'
    seimodel_file = '/archive/bioinformatics/Zhou_lab/shared/jzhou/cistrome/2019-07-25-13-21-44-IY28BMRY/best_model.pth.tar'

    ref_file = '/archive/bioinformatics/Zhou_lab/shared/jzhou/GraphSeq/Homo_sapiens.GRCh38.dna.primary_assembly.fa'
    ref_file_mmap = '/archive/bioinformatics/Zhou_lab/shared/jzhou/GraphSeq/Homo_sapiens.GRCh38.dna.primary_assembly.fa.mmap'
    tsses_file = '/archive/bioinformatics/Zhou_lab/shared/jzhou/FANTOM/analysis/FANTOM_CAT.lv3_robust.tss.sortedby_fantomcage.hg38.v4.tsv'

    diffusion_weights_file = '/archive/bioinformatics/Zhou_lab/shared/pavdeyev/notebooks/sde/golden_promoter/presampled_4C_maxt4.speed_balanced.100000.pth'

    device = 'cuda'
    batch_size = 256
    num_workers = 4

    n_time_steps = 400

    random_order = False
    speed_balanced = True

    ncat = 4

    num_epochs = 200

    lr = 5e-4


class GenomicSignalFeatures(Target):
    """
    #Accept a list of cooler files as input.
    """

    def __init__(self, input_paths, features, shape, blacklists=None, blacklists_indices=None,
                 replacement_indices=None, replacement_scaling_factors=None):
        """
        Constructs a new `GenomicFeatures` object.
        """
        self.input_paths = input_paths
        self.initialized = False
        self.blacklists = blacklists
        self.blacklists_indices = blacklists_indices
        self.replacement_indices = replacement_indices
        self.replacement_scaling_factors = replacement_scaling_factors

        self.n_features = len(features)
        self.feature_index_dict = dict(
            [(feat, index) for index, feat in enumerate(features)])
        self.shape = (len(input_paths), *shape)

    def get_feature_data(self, chrom, start, end, nan_as_zero=True, feature_indices=None):
        if not self.initialized:
            self.data = [pyBigWig.open(path) for path in self.input_paths]
            if self.blacklists is not None:
                self.blacklists = [tabix.open(blacklist) for blacklist in self.blacklists]
            self.initialized = True

        if feature_indices is None:
            feature_indices = np.arange(len(self.data))

        wigmat = np.zeros((len(feature_indices), end - start), dtype=np.float32)
        for i in feature_indices:
            try:
                wigmat[i, :] = self.data[i].values(chrom, start, end, numpy=True)
            except:
                print(chrom, start, end, self.input_paths[i], flush=True)
                raise

        if self.blacklists is not None:
            if self.replacement_indices is None:
                if self.blacklists_indices is not None:
                    for blacklist, blacklist_indices in zip(self.blacklists, self.blacklists_indices):
                        for _, s, e in blacklist.query(chrom, start, end):
                            wigmat[blacklist_indices, np.fmax(int(s) - start, 0): int(e) - start] = 0
                else:
                    for blacklist in self.blacklists:
                        for _, s, e in blacklist.query(chrom, start, end):
                            wigmat[:, np.fmax(int(s) - start, 0): int(e) - start] = 0
            else:
                for blacklist, blacklist_indices, replacement_indices, replacement_scaling_factor in zip(
                        self.blacklists, self.blacklists_indices, self.replacement_indices,
                        self.replacement_scaling_factors):
                    for _, s, e in blacklist.query(chrom, start, end):
                        wigmat[blacklist_indices, np.fmax(int(s) - start, 0): int(e) - start] = wigmat[
                                                                                                replacement_indices,
                                                                                                np.fmax(int(s) - start,
                                                                                                        0): int(
                                                                                                    e) - start] * replacement_scaling_factor

        if nan_as_zero:
            wigmat[np.isnan(wigmat)] = 0
        return wigmat


class TSSDatasetS(Dataset):
    def __init__(self, seqlength=1024, split="train", n_tsses=100000, rand_offset=0):
        self.shuffle = False

        self.genome = MemmapGenome(
            input_path='/archive/bioinformatics/Zhou_lab/shared/jzhou/GraphSeq/Homo_sapiens.GRCh38.dna.primary_assembly.fa',
            memmapfile='/archive/bioinformatics/Zhou_lab/shared/jzhou/GraphSeq/Homo_sapiens.GRCh38.dna.primary_assembly.fa.mmap',
            blacklist_regions='hg38'
        )
        self.tfeature = GenomicSignalFeatures(
            ["/archive/bioinformatics/Zhou_lab/shared/jzhou/FANTOM/agg.plus.bw.bedgraph.bw",
             "/archive/bioinformatics/Zhou_lab/shared/jzhou/FANTOM/agg.minus.bw.bedgraph.bw"],
            ['cage_plus', 'cage_minus'],
            (2000,),
            ["/archive/bioinformatics/Zhou_lab/shared/jzhou/FANTOM/fantom.blacklist8.plus.bed.gz",
             "/archive/bioinformatics/Zhou_lab/shared/jzhou/FANTOM/fantom.blacklist8.minus.bed.gz"])

        self.tsses = pd.read_table(
            '/archive/bioinformatics/Zhou_lab/shared/jzhou/FANTOM/analysis/FANTOM_CAT.lv3_robust.tss.sortedby_fantomcage.hg38.v4.tsv',
            sep='\t')
        self.tsses = self.tsses.iloc[:n_tsses, :]

        self.chr_lens = self.genome.get_chr_lens()
        self.split = split
        if split == "train":
            self.tsses = self.tsses.iloc[~np.isin(self.tsses['chr'].values, ['chr8', 'chr9', 'chr10'])]
        elif split == "valid":
            self.tsses = self.tsses.iloc[np.isin(self.tsses['chr'].values, ['chr10'])]
        elif split == "test":
            self.tsses = self.tsses.iloc[np.isin(self.tsses['chr'].values, ['chr8', 'chr9'])]
        else:
            raise ValueError
        self.rand_offset = rand_offset
        self.seqlength = seqlength

    def __len__(self):
        return self.tsses.shape[0]

    def __getitem__(self, tssi):
        chrm, pos, strand = self.tsses['chr'].values[tssi], self.tsses['TSS'].values[tssi], self.tsses['strand'].values[
            tssi]
        offset = 1 if strand == '-' else 0

        offset = offset + np.random.randint(-self.rand_offset, self.rand_offset + 1)
        seq = self.genome.get_encoding_from_coords(chrm, pos - int(self.seqlength / 2) + offset,
                                                   pos + int(self.seqlength / 2) + offset, strand)

        signal = self.tfeature.get_feature_data(chrm, pos - int(self.seqlength / 2) + offset,
                                                pos + int(self.seqlength / 2) + offset)
        if strand == '-':
            signal = signal[::-1, ::-1]
        return np.concatenate([seq, signal.T], axis=-1).astype(np.float32)

    def reset(self):
        np.random.seed(0)


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

    def __init__(self, embed_dim=256, time_dependent_weights=None, time_step=0.01):
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
        n = 256
        self.linear = nn.Conv1d(5, n, kernel_size=9, padding=4)
        self.blocks = nn.ModuleList([nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256)])

        self.denses = nn.ModuleList([Dense(embed_dim, n) for _ in range(20)])
        self.norms = nn.ModuleList([nn.GroupNorm(1, n) for _ in range(20)])

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.scale = nn.Parameter(torch.ones(1))
        self.final = nn.Sequential(nn.Conv1d(n, n, kernel_size=1),
                                   nn.GELU(),
                                   nn.Conv1d(n, 4, kernel_size=1))
        self.register_buffer("time_dependent_weights", time_dependent_weights)
        self.time_step = time_step

    def forward(self, x, t, t_ind=None, return_a=False):
        # Obtain the Gaussian random feature embedding for t
        # embed: [N, embed_dim]
        embed = self.act(self.embed(t / 2))

        # Encoding path
        # x: NLC -> NCL
        out = x.permute(0, 2, 1)
        out = self.act(self.linear(out))

        # pos encoding
        for block, dense, norm in zip(self.blocks, self.denses, self.norms):
            h = self.act(block(norm(out + dense(embed)[:, :, None])))
            if h.shape == out.shape:
                out = h + out
            else:
                out = h

        out = self.final(out)

        out = out.permute(0, 2, 1)

        if self.time_dependent_weights is not None:
            t_step = (t / self.time_step) - 1
            w0 = self.time_dependent_weights[t_step.long()]
            w1 = self.time_dependent_weights[torch.clip(t_step + 1, max=len(self.time_dependent_weights) - 1).long()]
            out = out * (w0 + (t_step - t_step.floor()) * (w1 - w0))[:, None, None]

        out = out - out.mean(axis=-1, keepdims=True)
        return out


if __name__ == '__main__':
    config = ModelParameters()

    sb = UnitStickBreakingTransform()

    seifeatures = pd.read_csv(config.seifeatures_file, sep='|', header=None)

    sei = nn.DataParallel(NonStrandSpecific(Sei(4096, 21907)))
    sei.load_state_dict(torch.load(config.seimodel_file, map_location='cpu')['state_dict'])
    sei.cuda()

    ### LOAD WEIGHTS
    d = torch.load(config.diffusion_weights_file)
    for k in d:
        globals()[k] = d[k]

    alpha = torch.ones(config.ncat - 1).float()
    beta =  torch.arange(config.ncat - 1, 0, -1).float()

    ### TIME DEPENDENT WEIGHTS ###
    torch.set_default_dtype(torch.float32)

    train_set = TSSDatasetS(n_tsses=40000, rand_offset=10)
    data_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    time_dependent_cums = torch.zeros(config.n_time_steps).to(config.device)
    time_dependent_counts = torch.zeros(config.n_time_steps).to(config.device)

    avg_loss = 0.
    num_items = 0
    for i, x in enumerate(data_loader):
        x = x[..., :4]
        random_t = torch.randint(0, config.n_time_steps, (x.shape[0],))

        order = np.random.permutation(np.arange(config.ncat))
        if config.random_order:
            # perturbed_x, perturbed_x_grad = diffusion_fast_flatdirichlet(x[...,order], random_t, v_one, v_one_loggrad)
            perturbed_x, perturbed_x_grad = diffusion_factory(x[..., order], random_t, v_one, v_zero, v_one_loggrad,
                                                              v_zero_loggrad, alpha, beta)
            perturbed_x = perturbed_x[..., np.argsort(order)]
            perturbed_x_grad = perturbed_x_grad[..., np.argsort(order)]
        else:
            # perturbed_x, perturbed_x_grad = diffusion_fast_flatdirichlet(x, random_t, v_one, v_one_loggrad)
            perturbed_x, perturbed_x_grad = diffusion_factory(x, random_t, v_one, v_zero, v_one_loggrad, v_zero_loggrad,
                                                              alpha, beta)
        perturbed_x = perturbed_x.to(config.device)
        perturbed_x_grad = perturbed_x_grad.to(config.device)
        random_t = random_t.to(config.device)
        perturbed_v = sb._inverse(perturbed_x)

        order = np.random.permutation(np.arange(config.ncat))

        if config.random_order:
            perturbed_v = sb._inverse(perturbed_x[..., order], prevent_nan=True).detach()
        else:
            perturbed_v = sb._inverse(perturbed_x, prevent_nan=True).detach()

        time_dependent_counts[random_t] += 1
        if config.speed_balanced:
            s = 2 / (torch.ones(config.ncat - 1, device=config.device) + torch.arange(config.ncat - 1, 0, -1,
                                                                                      device=config.device).float())
        else:
            s = torch.ones(config.ncat - 1, device=config.device)

        if config.random_order:
            time_dependent_cums[random_t] += (perturbed_v * (1 - perturbed_v) * s[(None,) * (x.ndim - 1)] * (
                gx_to_gv(perturbed_x_grad[..., order], perturbed_x[..., order])) ** 2).view(x.shape[0], -1).mean(
                dim=1).detach()
        else:
            time_dependent_cums[random_t] += (perturbed_v * (1 - perturbed_v) * s[(None,) * (x.ndim - 1)] * (
                gx_to_gv(perturbed_x_grad, perturbed_x)) ** 2).view(x.shape[0], -1).mean(dim=1).detach()

        time_dependent_weights = time_dependent_cums / time_dependent_counts
        time_dependent_weights = time_dependent_weights / time_dependent_weights.mean()

    plt.plot(torch.sqrt(time_dependent_weights.cpu()))
    plt.savefig("timedependent_weight.png")

    #### PREPARE Valid DATASET
    valid_set = TSSDatasetS(split='valid', n_tsses=40000, rand_offset=0)
    valid_data_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0)
    valid_datasets = []
    for x in valid_data_loader:
        valid_datasets.append(x)

    validseqs = []
    for seq in valid_datasets:
        validseqs.append(seq[:, :, :4])
    validseqs = np.concatenate(validseqs, axis=0)

    with torch.no_grad():
        validseqs_pred = np.zeros((2915, 21907))
        for i in range(int(validseqs.shape[0] / 128)):
            validseq = validseqs[i * 128:(i + 1) * 128]
            validseqs_pred[i * 128:(i + 1) * 128] = sei(
                torch.cat([torch.ones((validseq.shape[0], 4, 1536)) * 0.25, torch.FloatTensor(validseq).transpose(1, 2),
                           torch.ones((validseq.shape[0], 4, 1536)) * 0.25], 2).cuda()).cpu().detach().numpy()
        validseq = validseqs[-128:]
        validseqs_pred[-128:] = sei(
            torch.cat([torch.ones((validseq.shape[0], 4, 1536)) * 0.25, torch.FloatTensor(validseq).transpose(1, 2),
                       torch.ones((validseq.shape[0], 4, 1536)) * 0.25], 2).cuda()).cpu().detach().numpy()
    validseqs_predh3k4me3 = validseqs_pred[:, seifeatures[1].str.strip().values == 'H3K4me3'].mean(axis=1)

    #### TRAINING CODE
    score_model = nn.DataParallel(ScoreNet(time_dependent_weights=torch.sqrt(time_dependent_weights)))
    score_model = score_model.to(config.device)
    score_model.train()

    train_set = TSSDatasetS(n_tsses=40000, rand_offset=100)
    data_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    sampler = Euler_Maruyama_sampler

    optimizer = Adam(score_model.parameters(), lr=config.lr)

    torch.set_default_dtype(torch.float32)
    bestsei_validloss = float('Inf')

    tqdm_epoch = tqdm.trange(config.num_epochs)
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        stime = time.time()

        for xS in data_loader:
            x = xS[:, :, :4]
            s = xS[:, :, 4:5]

            # Optional : there are several options for importance sampling here. it needs to match the loss function
            random_t = torch.LongTensor(np.random.choice(np.arange(config.n_time_steps), size=x.shape[0],
                                                         p=(torch.sqrt(time_dependent_weights) / torch.sqrt(
                                                             time_dependent_weights).sum()).cpu().detach().numpy())).to(
                config.device)
            if config.random_order:
                order = np.random.permutation(np.arange(C))
                # perturbed_x, perturbed_x_grad = diffusion_fast_flatdirichlet(x[...,order], random_t, v_one, v_one_loggrad)
                perturbed_x, perturbed_x_grad = diffusion_factory(x[..., order], random_t, v_one, v_zero, v_one_loggrad,
                                                                  v_zero_loggrad, alpha, beta)

                perturbed_x = perturbed_x[..., np.argsort(order)]
                perturbed_x_grad = perturbed_x_grad[..., np.argsort(order)]
            else:
                perturbed_x, perturbed_x_grad = diffusion_fast_flatdirichlet(x, random_t, v_one, v_one_loggrad)
                # perturbed_x, perturbed_x_grad = diffusion_factory(x, random_t, v_one, v_zero, v_one_loggrad, v_zero_loggrad, alpha, beta)

            perturbed_x = perturbed_x.to(config.device)
            perturbed_x_grad = perturbed_x_grad.to(config.device)
            random_t = random_t.to(config.device)

            random_timepoints = timepoints[random_t]
            s = s.to(config.device)

            score = score_model(torch.cat([perturbed_x, s], -1), random_timepoints)

            # the loss weighting function may change, there are a few options that we will experiment on
            if config.speed_balanced:
                s = 2 / (torch.ones(config.ncat - 1, device=config.device) + torch.arange(config.ncat - 1, 0, -1,
                                                                                          device=config.device).float())
            else:
                s = torch.ones(config.ncat - 1, device=config.device)

            if config.random_order:
                order = np.random.permutation(np.arange(config.ncat))
                perturbed_v = sb._inverse(perturbed_x[..., order], prevent_nan=True).detach()
                loss = torch.mean(torch.mean(
                    1 / (torch.sqrt(time_dependent_weights))[random_t][(...,) + (None,) * (x.ndim - 1)] * s[
                        (None,) * (x.ndim - 1)] * perturbed_v * (1 - perturbed_v) * (
                                gx_to_gv(score[..., order], perturbed_x[..., order], create_graph=True) - gx_to_gv(
                            perturbed_x_grad[..., order], perturbed_x[..., order])) ** 2, dim=(1)))
            else:
                perturbed_v = sb._inverse(perturbed_x, prevent_nan=True).detach()
                loss = torch.mean(torch.mean(
                    1 / (torch.sqrt(time_dependent_weights))[random_t][(...,) + (None,) * (x.ndim - 1)] * s[
                        (None,) * (x.ndim - 1)] * perturbed_v * (1 - perturbed_v) * (
                                gx_to_gv(score, perturbed_x, create_graph=True) - gx_to_gv(perturbed_x_grad,
                                                                                           perturbed_x)) ** 2, dim=(1)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]

        # Print the averaged training loss so far.
        print(avg_loss / num_items)
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))

        if epoch % 5 == 0:
            score_model.eval()

            # generate sequence samples
            torch.set_default_dtype(torch.float32)
            allsamples = []
            for t in valid_datasets:
                allsamples.append(sampler(score_model,
                                          (1024, 4),
                                          batch_size=t.shape[0],
                                          max_time=4,
                                          min_time=4 / 400,
                                          time_dilation=1,
                                          num_steps=100,
                                          eps=1e-5,
                                          speed_balanced=config.speed_balanced,
                                          device=config.device,
                                          concat_input=t[:, :, 4:5].cuda()
                                          ).detach().cpu().numpy()
                                  )

            allsamples = np.concatenate(allsamples, axis=0)
            allsamples_pred = np.zeros((2915, 21907))
            for i in range(int(allsamples.shape[0] / 128)):
                seq = 1.0 * (allsamples[i * 128:(i + 1) * 128] > 0.5)
                allsamples_pred[i * 128:(i + 1) * 128] = sei(
                    torch.cat([torch.ones((seq.shape[0], 4, 1536)) * 0.25, torch.FloatTensor(seq).transpose(1, 2),
                               torch.ones((seq.shape[0], 4, 1536)) * 0.25], 2).cuda()).cpu().detach().numpy()
            seq = allsamples[-128:]
            allsamples_pred[-128:] = sei(
                torch.cat([torch.ones((seq.shape[0], 4, 1536)) * 0.25, torch.FloatTensor(seq).transpose(1, 2),
                           torch.ones((seq.shape[0], 4, 1536)) * 0.25], 2).cuda()).cpu().detach().numpy()

            allsamples_predh3k4me3 = allsamples_pred[:, seifeatures[1].str.strip().values == 'H3K4me3'].mean(axis=-1)
            valid_loss = ((validseqs_predh3k4me3 - allsamples_predh3k4me3) ** 2).mean()
            print(f"{epoch} valid sei loss {valid_loss} {time.time() - stime}", flush=True)

            if valid_loss < bestsei_validloss:
                print('Best valid SEI loss!')
                bestsei_validloss = valid_loss
                torch.save(score_model.state_dict(), 'sdedna_promoter_revision.sei.bestvalid.pth')

            score_model.train()
