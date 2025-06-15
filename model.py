import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from torch.utils.data import Dataset
from CliffordGreen import GaussianPointDescriptor
from Bio.PDB import PDBParser


class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_size, descriptor_dim):
        super().__init__()
        assert isinstance(hidden_size, (list, tuple)) and len(hidden_size) == 3, f"hidden_size must be a 3‐element list, got {hidden_size}"
        self.net = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_size[2]),
            nn.ReLU(),
            nn.Linear(hidden_size[2], hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], hidden_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_size[0], 2 * z_dim)   # outputs both mu and raw sigma, concatenated
        )
        self.z_dim = z_dim
        self.descriptor_dim = descriptor_dim

    def forward(self, x):
        """
        x: [B, descriptor_dim] batch size and descriptor dimension
        returns: (mu, sigma) each [B, z_dim]
        """
        assert x.dim() == 2, f"Encoder expected a 2D tensor, got shape {tuple(x.shape)}"
        batch_size, feat = x.shape
        assert feat == self.descriptor_dim, f"Encoder expected input of shape [batch_size, {self.descriptor_dim}], but got {tuple(x.shape)}"
        enc_out = self.net(x)                 # [B, 2*z_dim]
        mu, log_sigma = enc_out.chunk(2, dim=-1)
        sigma = F.softplus(log_sigma) + 1e-6  # ensure positivity
        assert mu.shape == (batch_size, self.z_dim), f"Encoder output mu has shape {tuple(mu.shape)}, expected ({batch_size}, {self.z_dim})"
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, descriptor_dim):
        super().__init__()
        assert isinstance(hidden_dim, (list, tuple)) and len(hidden_dim) == 3, f"hidden_dim must be a 3‐element list, got {hidden_dim}"
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.ReLU(),
            nn.Linear(hidden_dim[2], 2 * descriptor_dim)  # outputs both mu and raw sigma concatenated
        )
        self.z_dim = z_dim
        self.descriptor_dim = descriptor_dim

    def forward(self, z):
        """
        :param z: [B, z_dim]
        :returns:
           mu    : [B, descriptor_dim]
           sigma : [B, descriptor_dim]
        """
        assert z.dim() == 2, f"Decoder expected 2D input, got shape {tuple(z.shape)}"
        B, zd = z.shape
        assert zd == self.z_dim, f"Decoder expected z of shape [{B}, {self.z_dim}], but got {tuple(z.shape)}"
        out = self.net(z)                   # [B, 2*D]
        mu, log_sigma = out.chunk(2, dim=-1)
        sigma = F.softplus(log_sigma) + 1e-6
        assert mu.shape == (B, self.descriptor_dim), f"Decoder output mu has shape {tuple(mu.shape)}, expected ({B}, {self.descriptor_dim})"
        return mu, sigma

class VAE(nn.Module):
    def __init__(self,
                 descriptor_dim: int,
                 z_dim: int,
                 hidden_size,
                 D_mean,
                 D_std):
        super().__init__()
        self.descriptor_dim = descriptor_dim
        self.z_dim = z_dim
        self.D_mean = D_mean
        self.D_std = D_std

        # Precompute diagonal vs off-diagonal indices
        all_idx = list(range(descriptor_dim))
        diag_idx     = [i for i in all_idx if self.is_diagonal(i)]
        off_diag_idx = [i for i in all_idx if not self.is_diagonal(i)]

        # Register as buffers so they move to CUDA correctly
        self.register_buffer('idx_diag', torch.LongTensor(diag_idx))
        self.register_buffer('idx_off',  torch.LongTensor(off_diag_idx))
        self.register_buffer('num_diag', torch.tensor(len(diag_idx)))
        self.register_buffer('num_off',  torch.tensor(len(off_diag_idx)))

        self.encoder = Encoder(z_dim, hidden_size, descriptor_dim)
        self.decoder = Decoder(z_dim, hidden_size, descriptor_dim)

    def model(self, x, beta=1.0, *args, **kwargs):
        """
        x: [B, descriptor_dim]
        """

        assert x.dim() == 2, f"model() got x shape {tuple(x.shape)}, expected 2D"
        B, Ddim = x.shape
        assert Ddim == self.descriptor_dim, f"model() expected descriptor_dim={self.descriptor_dim}, but got {Ddim}"

        pyro.module("decoder", self.decoder)
        z_loc = x.new_zeros(self.z_dim)
        z_scale = x.new_ones(self.z_dim)

        with pyro.plate("batch", B):
            # sample from the normal prior
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1), infer={"scale": beta})

            # decode into mu and sigma
            mu, sigma = self.decoder(z)  # each [B, descriptor_dim]

            # Off-diagonals (Normal)
            obs_off   = x.index_select(dim=1, index=self.idx_off)  # [B, K_off]
            mu_off    =    mu.index_select(dim=1, index=self.idx_off)  # [B, K_off]
            sigma_off = sigma.index_select(dim=1, index=self.idx_off)  # [B, K_off]

            pyro.sample("D_off", dist.Normal(mu_off, sigma_off).to_event(1), obs=obs_off)

            # Diagonals (Chi2)
            obs_diag = x.index_select(dim=1, index=self.idx_diag)   # [B, K_diag]
            D_std_d  = self.D_std.index_select(dim=0, index=self.idx_diag)  # [K_diag]
            D_mean_d = self.D_mean.index_select(dim=0, index=self.idx_diag) # [K_diag]

            raw = obs_diag * D_std_d.unsqueeze(0) + D_mean_d.unsqueeze(0)  # [B, K_diag]
            obs_sq = torch.clamp(raw * raw, min=1e-3)                      # [B, K_diag]

            pyro.sample("D_diag", dist.Chi2(df=3).expand([B, self.num_diag]).to_event(1), obs=obs_sq)

    def guide(self, x, *args, **kwargs):
        """
        x: [B, descriptor_dim]
        """
        B, Ddim = x.shape
        assert Ddim == self.descriptor_dim

        pyro.module("encoder", self.encoder)

        with pyro.plate("batch", B):
            mu_z, sigma_z = self.encoder(x)
            pyro.sample("latent", dist.Normal(mu_z, sigma_z).to_event(1))

    def is_diagonal(self, i):
        return (i == 0) or ((i - 1) % 2 == 0)

    def reconstruct_from_mean(self, D_norm):
        """
        Given a single normalized descriptor D_norm ([descriptor_dim]),
        encode to z, decode its mean back to a descriptor, un‐normalize,
        and reconstruct 3D coords.
        """
        # pack into a batch of size 1
        Dn = D_norm.unsqueeze(0)  # [1, descriptor_dim]
        # encode and get mean z
        z_mu, _ = self.encoder(Dn)  # both [1, z_dim]
        mu_D_norm, _ = self.decoder(z_mu)  # both [1, descriptor_dim]
        D_rec_raw = mu_D_norm.squeeze(0) * self.D_std + self.D_mean  # [descriptor_dim]
        X_rec = GaussianPointDescriptor.descriptor_to_coordinates(D_rec_raw)
        mse = (D_rec_raw - D_norm * self.D_std - self.D_mean).pow(2).mean().item()

        return X_rec, mse, z_mu


class CliffordGreenDescriptorDataset(Dataset):
    """
    Loads one multi‐model PDB file, extracts only backbone atoms (N, CA, C)
    from each model (in residue‐order) and builds Clifford–Green descriptors.
    """
    def __init__(self, pdb_path: str):
        parser     = PDBParser(QUIET=True)
        structure  = parser.get_structure("ensemble", pdb_path)
        coords_list = []

        # collect [N_bb × 3] arrays for each model
        for model in structure:
            bb_coords = []
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        if atom.get_name() in ("N", "CA", "C"):
                            bb_coords.append(atom.get_coord())
            if bb_coords:
                coords_list.append(np.stack(bb_coords, axis=0))

        if not coords_list:
            raise ValueError(f"No backbone atoms found in {pdb_path}")

        lengths = [c.shape[0] for c in coords_list]
        if len(set(lengths)) != 1:
            raise ValueError(f"Backbone atom counts differ across models: {set(lengths)}")
        self.N_bb = lengths[0]

        X_np = np.stack(coords_list, axis=0).astype(np.float32)
        self.X = torch.from_numpy(X_np)

        D_list = []
        for m in range(self.X.shape[0]):
            D = GaussianPointDescriptor.coordinates_to_descriptor(self.X[m])
            D_list.append(D)
        self.D = torch.stack(D_list, dim=0)

    def __len__(self):
        return self.D.size(0)

    def __getitem__(self, idx):
        # returns (descriptor, backbone_coords_of_that_model)
        return self.D[idx], self.X[idx]