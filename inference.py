import torch
import pyro
import numpy as np
import argparse
from pathlib import Path

from sklearn.decomposition import PCA

from model import VAE, load_all_atoms, save_all_atom_pdb, get_all_atom_line

# def generate_synthetic_structures(
#     vae: VAE,
#     info_list: list,
#     output_dir: str,
#     num_samples: int = 5,
#     device: torch.device = torch.device("cpu")
# ):
#     """
#     Sample `num_samples` synthetic structures from p(z)=N(0,I), decode to descriptors,
#     reconstruct 3D coords, and save each as “synthetic_<i>.pdb” in `output_dir`.
#
#     Arguments:
#         vae         : a trained VAE instance (already .to(device) and .eval()).
#         info_list   : the list of atom‐metadata (same length as number of atoms) you got
#                       from load_all_atoms(...) on some reference PDB.  Its only role is to
#                       carry chain/residue/atom names, so we can write PDB lines.
#         output_dir  : folder where “synthetic_0.pdb”, …, “synthetic_{num_samples-1}.pdb”
#                       will be written.  It will be created if it doesn’t exist.
#         num_samples : how many independent draws from N(0,I) you want.
#         device      : torch device (“cpu” or “cuda”) to run on.
#
#     Returns:
#         A list of length `num_samples`, where each entry is a tensor of shape
#         [N_atoms × 3] containing the reconstructed coordinates.
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     desc_dim = vae.descriptor_dim
#
#     synthetic_coords_list = []
#
#     with torch.no_grad():
#         for i in range(num_samples):
#             # 1) Sample z ~ N(0, I) of shape [1 × z_dim]
#             z = torch.randn(1, vae.z_dim, device=device)
#
#             # 2) Decode into a 1×(2*desc_dim) tensor
#             all_params = vae.decoder(z)            # [1, 2*desc_dim]
#
#             # 3) Discard the second half (log-σ) and keep first half = μ_norm
#             half    = all_params.size(1) // 2      # = desc_dim
#             mu_norm = all_params[:, :half]         # shape [1, desc_dim]
#
#             # 4) Un-normalize back to raw descriptor
#             D_recon_raw = mu_norm.squeeze(0) * vae.D_std + vae.D_mean   # [desc_dim]
#
#             # 5) Reconstruct 3D coordinates (N_atoms × 3)
#             X_recon = GaussianPointDescriptor.descriptor_to_coordinates(D_recon_raw)  # [N_atoms, 3]
#
#             synthetic_coords_list.append(X_recon.cpu())
#
#             # 6) Write out a new PDB file “synthetic_<i>.pdb”
#             out_path = os.path.join(output_dir, f"synthetic_{i}.pdb")
#             save_all_atom_pdb(X_recon.cpu(), info_list, out_path)
#
#     return synthetic_coords_list

import torch
import numpy as np
from CliffordGreen import GaussianPointDescriptor

def interpolate_models(
    vae,
    pdb1: str,
    pdb2: str,
    n_steps: int = 10,
    device: torch.device = torch.device("cpu"),
    out_prefix: str = "interp"
):
    """
    1) Load two PDBs ➔ X1,X2 [N_atoms×3]
    2) Compute descriptors D1,D2 (and normalize)
    3) Encode to latent means z1,z2
    4) For t in [0..1], zt = (1−t)·z1 + t·z2
       • Decode zt ➔ D̂t (normalized descriptor)
       • Un-normalize ➔ Dt
       • Reconstruct Xt via descriptor_to_coordinates
       • Write out MODEL #t
    """
    vae.eval()
    # (1) load and descriptor‐ify
    X1, info1 = load_all_atoms(pdb1)
    X2, info2 = load_all_atoms(pdb2)
    assert X1.shape==X2.shape, "must have same atom count"
    D1 = GaussianPointDescriptor.coordinates_to_descriptor(X1.to(device))
    D2 = GaussianPointDescriptor.coordinates_to_descriptor(X2.to(device))
    # normalize with the same mean/std used in training
    D1n = (D1 - vae.D_mean) / vae.D_std
    D2n = (D2 - vae.D_mean) / vae.D_std

    # (3) encode to latent means
    with torch.no_grad():
        μ1, _ = vae.encoder(D1n.unsqueeze(0))   # [1,z_dim]
        μ2, _ = vae.encoder(D2n.unsqueeze(0))

    # now open an output PDB
    fp = open(f"{out_prefix}.pdb","w")
    for i,α in enumerate(np.linspace(0,1,n_steps)):
        zt = (1-α)*μ1 + α*μ2                    # [1,z_dim]
        # decode to normalized descriptor
        all_out = vae.decoder(zt)               # [1,2*desc_dim]
        mu_norm, _ = all_out.chunk(2,dim=-1)    # split off μ
        Dt_n = mu_norm.squeeze(0)               # [desc_dim]
        # un-normalize
        Dt   = Dt_n*vae.D_std + vae.D_mean
        # reconstruct 3D coords
        Xt   = GaussianPointDescriptor.descriptor_to_coordinates(Dt)
        # write as model
        fp.write(f"MODEL     {i+1}\n")
        for idx, atom_info in enumerate(info1):
            x,y,z = Xt[idx].tolist()
            line = get_all_atom_line(
                idx+1,
                atom_info['atom_name'],
                atom_info['res_name'],
                atom_info['chain_id'],
                atom_info['res_id'],
                atom_info['altloc'],
                atom_info['i_code'],
                x,y,z,
                atom_info['occupancy'],
                atom_info['tempfactor'],
                atom_info['element'],
                atom_info['charge']
            )
            fp.write(line)
        fp.write("ENDMDL\n")
    fp.close()
    print(f"Wrote interpolation PDB: {out_prefix}.pdb")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # #Load ALL ATOMS from the input PDB
    # X_orig, info = load_all_atoms("data/1fsd.pdb")
    # print(f"Loaded {X_orig.shape[0]} atoms from {"data/1fsd.pdb"}")
    #
    # #Convert to a Clifford–Green  descriptor D
    # D = GaussianPointDescriptor.coordinates_to_descriptor(X_orig)
    # print(f"Descriptor length = {D.shape[0]}")
    #
    # #Reconstruct coordinates X_recon from D
    # X_recon = GaussianPointDescriptor.descriptor_to_coordinates(D)
    # print("Reconstructed coordinates shape:", X_recon.shape)
    #
    # #Write out a new PDB with the same atom ordering/annotations, but new coords
    # save_all_atom_pdb(X_recon, info, "data/reconstructed.pdb")
    # print(f"Reconstructed PDB written to: reconstructed.pdb")
    # exit()

    ckpt = torch.load("vae_checkpoint.pt", map_location=device)

    desc_dim = ckpt["desc_dim"]  # int
    D_mean = ckpt["D_mean"].to(device)  # tensor [desc_dim]
    D_std = ckpt["D_std"].to(device)  # tensor [desc_dim]
    state = ckpt["state_dict"]

    vae = VAE(
        descriptor_dim=desc_dim,
        z_dim=4,
        hidden_size=64,
        D_mean=D_mean,
        D_std=D_std
    ).to(device)
    vae.load_state_dict(state)
    vae.eval()


    new_pdb = args.input_pdb
    X_new, info_new = load_all_atoms(new_pdb)

    D_new = GaussianPointDescriptor.coordinates_to_descriptor(X_new).to(device)

    if D_new.shape[0] != desc_dim:
        raise RuntimeError(
            f"Descriptor‐length mismatch: trained on {desc_dim}, but new PDB gave {D_new.shape[0]}"
        )

    D_norm = (D_new - D_mean) / D_std
    D_batch = D_norm.unsqueeze(0)

    with torch.no_grad():
        mu_z, sigma_z = vae.encoder(D_batch)
        z = mu_z

        all_params = vae.decoder(z)
        print("Expected 2*descriptor_dim:", 2 * desc_dim)

        half    = all_params.size(1) // 2
        print("Splitting at index:", half)
        mu_norm = all_params[:, :half]
        print("mu_norm.shape:", mu_norm.shape)

        D_recon_raw = mu_norm.squeeze(0) * D_std + D_mean
        print("D_recon_raw.shape:", D_recon_raw.shape)

    X_recon = GaussianPointDescriptor.descriptor_to_coordinates(D_recon_raw)

    save_all_atom_pdb(X_recon, info_new, args.output_pdb)
    print("Wrote reconstructed.pdb")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_pdb",    type=str, default="data/1fsd.pdb")
    p.add_argument("--output_pdb",   type=str, default="data/reconstruct.pdb")
    args = p.parse_args()
    main(args)