import torch
import argparse
from CliffordGreen import GaussianPointDescriptor

from model import VAE
from pdb_handler import load_backbone_atoms, save_backbone_pdb

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load("vae_checkpoint.pt", map_location=device)

    desc_dim = ckpt["desc_dim"]
    D_mean = ckpt["D_mean"].to(device)
    D_std = ckpt["D_std"].to(device)
    state = ckpt["state_dict"]

    vae = VAE(
        descriptor_dim=desc_dim,
        z_dim=8,
        hidden_size=[64, 128, 256],
        D_mean=D_mean,
        D_std=D_std
    ).to(device)
    vae.load_state_dict(state)
    vae.eval()


    new_pdb = args.input_pdb
    X_new, info_new = load_backbone_atoms(new_pdb)

    D_new = GaussianPointDescriptor.coordinates_to_descriptor(X_new).to(device)

    if D_new.shape[0] != desc_dim:
        raise RuntimeError(
            f"Descriptor‐length mismatch: trained on {desc_dim}, but new PDB gave {D_new.shape[0]}"
        )

    D_norm = (D_new - D_mean) / D_std
    D_batch = D_norm.unsqueeze(0)

    with torch.no_grad():
        # encode to latent
        mu_z, sigma_z = vae.encoder(D_batch)
        z = mu_z
        # decode directly to (mu, sigma)
        mu_norm, _ = vae.decoder(z)  # [1, descriptor_dim]
        D_recon_raw = mu_norm.squeeze(0) * D_std + D_mean  # [descriptor_dim]

        X_recon = GaussianPointDescriptor.descriptor_to_coordinates(D_recon_raw)

    save_backbone_pdb(X_recon, info_new, args.output_pdb)
    print("Wrote " + args.output_pdb)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_pdb", type=str, default="data/1unc.pdb")
    p.add_argument("--output_pdb", type=str, default="data/reconstruct.pdb")
    args = p.parse_args()
    main(args)