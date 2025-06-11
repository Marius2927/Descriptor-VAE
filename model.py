import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from torch.utils.data import Dataset, DataLoader, random_split
from CliffordGreen import GaussianPointDescriptor
import Bio.PDB as bio
from Bio.PDB import PDBParser

def g(x):
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))

def log_g(x):
    return torch.where(x >= 0, (x + 0.5).log(), -F.softplus(-x))

class MiniGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_z = nn.Linear(input_size, hidden_size)
        self.linear_h = nn.Linear(input_size, hidden_size)

    def forward(self, x_t, h_prev):
        z = torch.sigmoid(self.linear_z(x_t))
        h_tilde = g(self.linear_h(x_t))
        h_t = (1 - z) * h_prev + z * h_tilde
        return h_t

class MiniGRU_parallel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_z = nn.Linear(input_size, hidden_size)
        self.linear_h = nn.Linear(input_size, hidden_size)

    def forward_parallel(self, x, h0):
        seq_len, _ = x.shape
        k = self.linear_z(x)
        log_z = -F.softplus(-k)
        log_coeffs = -F.softplus(k)
        log_h0 = log_g(h0)
        log_tilde_h = log_g(self.linear_h(x))
        sequence = torch.cat([log_h0.unsqueeze(0), log_z + log_tilde_h], dim=0)
        a_star = F.pad(torch.cumsum(log_coeffs, dim=0), (0,0,1,0))
        log_h0_plus_b_star = torch.logcumsumexp(sequence - a_star, dim=0)
        log_h = a_star + log_h0_plus_b_star
        return torch.exp(log_h)[1:]

class Encoder(nn.Module):
    def __init__(self, descriptor_dim, z_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * z_dim)
        )
        self.z_dim = z_dim

    def forward(self, x):
        """
        :param x: [B, descriptor_dim]
        :returns: (mu, sigma) each of shape [B, z_dim]
        """
        enc_out = self.net(x)            # [B, 2*z_dim]
        mu, log_sigma = enc_out.chunk(2, dim=-1)
        sigma = F.softplus(log_sigma) + 1e-6
        return mu, sigma

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_size, descriptor_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * descriptor_dim)
        )

    def forward(self, z):
        """
        :param z: [B, z_dim]
        :returns: concatenated [mu | log_sigma] of shape [B, 2*descriptor_dim]
        """
        return self.net(z)

class VAE(nn.Module):
    def __init__(self,
                 descriptor_dim: int,
                 z_dim: int,
                 hidden_size: int,
                 D_mean: torch.Tensor,
                 D_std: torch.Tensor):
        super().__init__()
        self.descriptor_dim = descriptor_dim
        self.z_dim = z_dim
        self.D_mean = D_mean   # [descriptor_dim]
        self.D_std  = D_std    # [descriptor_dim]

        # Precompute diagonal vs off-diagonal indices
        all_idx = list(range(descriptor_dim))
        diag_idx     = [i for i in all_idx if self.is_diagonal(i)]
        off_diag_idx = [i for i in all_idx if not self.is_diagonal(i)]

        # Register as buffers so they move to CUDA correctly
        self.register_buffer('idx_diag', torch.LongTensor(diag_idx))
        self.register_buffer('idx_off',  torch.LongTensor(off_diag_idx))
        self.register_buffer('num_diag', torch.tensor(len(diag_idx)))
        self.register_buffer('num_off',  torch.tensor(len(off_diag_idx)))

        self.encoder = Encoder(descriptor_dim, z_dim, hidden_size)
        self.decoder = Decoder(z_dim, hidden_size, descriptor_dim)

    def model(self, x, beta=1.0, *args, **kwargs):
        """
        x: [B, descriptor_dim]
        """
        B, Ddim = x.shape
        assert Ddim == self.descriptor_dim

        pyro.module("decoder", self.decoder)
        z_loc   = x.new_zeros(self.z_dim)
        z_scale = x.new_ones(self.z_dim)

        with pyro.plate("batch", B):
            z = pyro.sample("latent",
                            dist.Normal(z_loc, z_scale).to_event(1), infer={"scale": beta})

            out = self.decoder(z)
            mu, log_sigma = out.chunk(2, dim=-1)
            sigma = F.softplus(log_sigma) + 1e-6

            # Off-diagonals (Normal)
            obs_off   = x.index_select(dim=1, index=self.idx_off)  # [B, K_off]
            mu_off    =    mu.index_select(dim=1, index=self.idx_off)  # [B, K_off]
            sigma_off = sigma.index_select(dim=1, index=self.idx_off)  # [B, K_off]

            pyro.sample("D_off",
                        dist.Normal(mu_off, sigma_off).to_event(1),
                        obs=obs_off)

            # Diagonals (Chi2)
            obs_diag = x.index_select(dim=1, index=self.idx_diag)   # [B, K_diag]
            D_std_d  = self.D_std.index_select(dim=0, index=self.idx_diag)  # [K_diag]
            D_mean_d = self.D_mean.index_select(dim=0, index=self.idx_diag) # [K_diag]

            raw = obs_diag * D_std_d.unsqueeze(0) + D_mean_d.unsqueeze(0)  # [B, K_diag]
            obs_sq = torch.clamp(raw * raw, min=1e-3)                      # [B, K_diag]

            pyro.sample("D_diag",
                        dist.Chi2(df=3).expand([B, self.num_diag]).to_event(1),
                        obs=obs_sq)

    def guide(self, x, *args, **kwargs):
        """
        x: [B, descriptor_dim]
        """
        B, Ddim = x.shape
        assert Ddim == self.descriptor_dim

        pyro.module("encoder", self.encoder)

        with pyro.plate("batch", B):
            mu_z, sigma_z = self.encoder(x)
            pyro.sample("latent",
                        dist.Normal(mu_z, sigma_z).to_event(1))

    def is_diagonal(self, i):
        return (i == 0) or ((i - 1) % 2 == 0)

    def reconstruct(self, x):
        """
        Reconstruct 3D coordinates from input descriptor x by sampling z.
        x: [seq_len, descriptor_dim]
        Returns: X_rec: [N, 3]
        """

        z_loc, z_sigma = self.encoder(x)
        z = dist.Normal(z_loc, z_sigma).sample()
        seq_len, descriptor_dim = x.shape

        out = self.decoder(z, seq_len)
        mu, log_sigma = out.chunk(2, dim=-1)

        D_rec = mu.squeeze(0)

        X_rec = GaussianPointDescriptor.descriptor_to_coordinates(D_rec, torch.eye(3, device=D_rec.device, dtype=D_rec.dtype))
        return X_rec

    def reconstruct_from_mean(self, x):
        """
        Reconstruct 3D coordinates using the mean latent vector.
        x: [seq_len, descriptor_dim]
        Returns: X_rec: [N, 3]
        """
        z_loc, z_sigma = self.encoder(x)
        z = z_loc  # use mean
        seq_len, descriptor_dim = x.shape
        out = self.decoder(z, seq_len)
        mu, log_sigma = out.chunk(2, dim=-1)
        D_rec = mu.squeeze(0)
        X_rec = GaussianPointDescriptor.descriptor_to_coordinates(D_rec, torch.eye(3, device=D_rec.device, dtype=D_rec.dtype))
        return X_rec


class CliffordGreenDescriptorDataset(Dataset):
    def __init__(self, pdb_files):
        parser = PDBParser(QUIET=True)
        all_coords = []
        for pdb_path in pdb_files:
            structure = parser.get_structure("prot", pdb_path)
            for model in structure.get_models():
                coords = [atom.get_coord()
                          for atom in model.get_atoms()]
                if coords:
                    all_coords.append(np.stack(coords, axis=0))

        # keep only those with the most common atom‐count
        N_atoms_list = [c.shape[0] for c in all_coords]
        mode_N       = max(set(N_atoms_list), key=N_atoms_list.count)
        kept = [c for c in all_coords if c.shape[0] == mode_N]

        X_np = np.stack(kept, axis=0).astype(np.float32)
        self.X = torch.from_numpy(X_np)

        D_list = []
        for b in range(len(self.X)):
            D_list.append(GaussianPointDescriptor.coordinates_to_descriptor(self.X[b]))
        self.D = torch.stack(D_list, dim=0)

    def __len__(self):
        return self.D.shape[0]

    def __getitem__(self, idx):
        return self.D[idx], self.X[idx]


def setup_data_loaders(pdb_files, train_ratio=0.8):
    dataset = CliffordGreenDescriptorDataset(pdb_files)
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, test_loader

def load_all_atoms(pdb_path: str):
    """
    Parse the input PDB file and return:
      - X         : torch.Tensor of shape [N_atoms, 3], in the exact order atoms appear
      - info_list : list of length N_atoms, where each entry is a dict containing:
            {
              'atom_name': str,
              'res_name' : str,
              'chain_id' : str,
              'res_id'   : int,
              'altloc'   : str,
              'i_code'   : str,
              'occupancy': float,
              'tempfactor': float,
              'element'  : str,
              'charge'   : str,
            }
    Loads both ATOM and HETATM records.
    """
    parser    = bio.PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)
    model     = next(structure.get_models())

    atom_coords = []
    info_list   = []

    for chain in model:
        chain_id = chain.get_id()
        for residue in chain:
            hetflag, resseq, icode = residue.get_id()
            res_name = residue.get_resname()
            for atom in residue:
                coord = atom.get_coord().astype(np.float32)
                atom_name = atom.get_name()
                altloc    = atom.get_altloc()
                element   = atom.element
                occ       = atom.get_occupancy() if atom.get_occupancy() is not None else 1.00
                tfac      = atom.get_bfactor()   if atom.get_bfactor()   is not None else 0.00
                charge    = atom.get_fullname().strip()[-2:].strip()

                atom_coords.append(coord)
                info_list.append({
                    'atom_name' : atom_name,
                    'res_name'  : res_name,
                    'chain_id'  : chain_id,
                    'res_id'    : resseq,
                    'altloc'    : altloc,
                    'i_code'    : icode,
                    'occupancy' : occ,
                    'tempfactor': tfac,
                    'element'   : element.strip(),
                    'charge'    : charge
                })

    if len(atom_coords) == 0:
        raise ValueError(f"No atom records found in {pdb_path}. Please check the file path or its contents.")

    X_np = np.vstack(atom_coords)
    X    = torch.from_numpy(X_np)
    return X, info_list


_ATOM_LINE_FMT = "%-6s%5d %4s%1s%3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s\n"

def get_all_atom_line(idx_atom: int,
                      atom_name: str,
                      res_name: str,
                      chain_id: str,
                      res_id: int,
                      altloc: str,
                      i_code: str,
                      x: float, y: float, z: float,
                      occupancy: float,
                      tempfactor: float,
                      element: str,
                      charge: str) -> str:
    record   = "ATOM"
    atom_num = idx_atom
    atom_name_field = atom_name.rjust(4)[:4]
    res_name_field  = res_name.ljust(3)[:3]
    chain_field     = chain_id[:1]
    i_code_field    = i_code[:1]
    element_field   = element.strip().rjust(2)[:2]
    charge_field    = charge.strip().rjust(2)[:2]

    return _ATOM_LINE_FMT % (
        record,
        atom_num,
        atom_name_field,
        altloc[:1],
        res_name_field,
        chain_field,
        res_id,
        i_code_field,
        x, y, z,
        occupancy,
        tempfactor,
        element_field,
        charge_field
    )


def save_all_atom_pdb(X: torch.Tensor,
                      info_list: list,
                      out_path: str):
    assert X.shape[0] == len(info_list), "Mismatch between X rows and info_list entries."

    with open(out_path, "w") as fp:
        fp.write("MODEL        1\n")
        for idx in range(len(info_list)):
            info = info_list[idx]
            x, y, z = X[idx].tolist()
            line = get_all_atom_line(
                idx_atom   = idx+1,
                atom_name  = info['atom_name'],
                res_name   = info['res_name'],
                chain_id   = info['chain_id'],
                res_id     = info['res_id'],
                altloc     = info['altloc'],
                i_code     = info['i_code'],
                x=x, y=y, z=z,
                occupancy  = info['occupancy'],
                tempfactor = info['tempfactor'],
                element    = info['element'],
                charge     = info['charge']
            )
            fp.write(line)
        fp.write("ENDMDL\n")