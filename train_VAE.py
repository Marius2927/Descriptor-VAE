import argparse
import random
import torch
import numpy as np
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from model import VAE, CliffordGreenDescriptorDataset

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
pyro.set_rng_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
Z_DIM = 8
HIDDEN_SIZE = [64, 128, 256]
LEARNING_RATE = 1e-6
NUM_EPOCHS = 10000
WARMUP = 1000

def train(svi, train_loader, beta, use_cuda=False):
    epoch_loss = 0.0
    for D_batch, X_batch in train_loader:
        D_batch = D_batch.to(device)
        X_batch = X_batch.to(device)
        epoch_loss += svi.step(D_batch, X_batch, beta)
    normalizer_train = len(train_loader.dataset)
    return epoch_loss / normalizer_train

def evaluate(svi, test_loader, use_cuda=False):
    total_loss = 0.0
    for D_batch, X_batch in test_loader:
        D_batch = D_batch.to(device)
        X_batch = X_batch.to(device)
        total_loss += svi.evaluate_loss(D_batch, X_batch)
    normalizer_test = len(test_loader.dataset)
    return total_loss / normalizer_test

def kabsch_rmsd(A, B):
    A0 = A - A.mean(axis=0, keepdims=True)
    B0 = B - B.mean(axis=0, keepdims=True)
    C  = A0.T @ B0

    U, S, Vt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1,1,d]) @ U.T

    A1 = A0 @ R.T
    diff2 = (A1 - B0)**2
    return np.sqrt(diff2.mean())

def main(args):
    Z_DIM = args.z_dim
    HIDDEN_SIZE = args.hidden_sizes
    LEARNING_RATE = args.learning_rate
    NUM_EPOCHS = args.num_epochs
    pdb_file = args.pdb
    pyro.clear_param_store()

    dataset = CliffordGreenDescriptorDataset(pdb_file)

    descriptor_dim = dataset.D.shape[1]
    D_mean = dataset.D.mean(dim=0)
    D_std = dataset.D.std(dim=0) + 1e-6

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    vae = VAE(descriptor_dim, z_dim=Z_DIM, hidden_size=HIDDEN_SIZE, D_mean=D_mean, D_std=D_std)
    opt = Adam({"lr": LEARNING_RATE})
    svi = SVI(vae.model, vae.guide, opt, loss=Trace_ELBO())

    train_elbo = []
    train_mse = []
    train_rmsd = []
    zs = []
    latent_hist = []

    patience = 25  # stop if no improvement for 25 epochs
    epochs_without_improve = 0
    best_rmsd = float('inf')
    for epoch in range(NUM_EPOCHS):
        beta = min(1.0, epoch / WARMUP)
        avg_elbo = train(svi, train_loader, beta)
        train_elbo.append(-avg_elbo)

        rmsd_accum = 0.0
        n_samples = 0
        mse_accum = 0
        with torch.no_grad():
            for D_norm, X_true in train_loader:
                D_norm = D_norm.to(device)
                X_true = X_true.to(device)
                B, _ = D_norm.shape

                for b in range(B):
                    X_rec, mse, mu_z = vae.reconstruct_from_mean(D_norm[b])
                    rmsd_accum += kabsch_rmsd(
                        X_true[b].cpu().numpy(),
                        X_rec.cpu().numpy()
                    )
                    mse_accum += mse
                    zs.append(mu_z.cpu())
                n_samples += B
        current_rmsd = rmsd_accum / n_samples
        train_mse.append(mse_accum / n_samples)
        train_rmsd.append(rmsd_accum / n_samples)
        latent_hist.append(torch.cat(zs, dim=0).view(-1).numpy())

        if current_rmsd<best_rmsd:
            best_rmsd = current_rmsd

        elif epoch>WARMUP:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(f"Early stopping at epoch {epoch} (best RMSD: {train_rmsd[-patience]:.4f})")
                break

        print(f"Epoch {epoch:3d} | ELBO={-avg_elbo:8.3f}| MSE={train_mse[-1]:5.3f} | RMSD={best_rmsd:5.3f}")

    torch.save({
        "desc_dim": descriptor_dim,
        "D_mean": D_mean.cpu(),
        "D_std": D_std.cpu(),
        "state_dict": vae.state_dict()
    }, "vae_checkpoint.pt")

    plt.figure()
    plt.plot(train_elbo, label="Train ELBO")
    plt.xlabel("Epoch")
    plt.ylabel("ELBO")
    plt.legend()
    plt.savefig("plt1.png")

    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.plot(train_mse, label="Descriptor MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_rmsd, label="3D-RMSD")
    plt.xlabel("Epoch")
    plt.ylabel("RMSD (Å)")
    plt.legend()
    plt.savefig("plt2.png")

    for i, z_all in enumerate(latent_hist[::max(1, NUM_EPOCHS // 5)]):
        plt.figure()
        plt.hist(z_all, bins=50, density=True)
        plt.title(f"Epoch {i * (NUM_EPOCHS // 5)} latent z distribution")
        plt.xlabel("z")
        plt.ylabel("Density")
        plt.savefig("plt"+str(i+3)+".png")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pdb", type=str, default='data/1unc.pdb')
    p.add_argument("--z_dim", choices=[4, 8, 16, 32], default=8)
    p.add_argument("--hidden_sizes", choices=[[32, 64, 128], [64, 128, 256]], default=[64, 128, 256])
    p.add_argument("--num_epochs", choices=[500, 1000, 5000, 10000], default=2000)
    p.add_argument("--learning_rate", choices=[1e-5, 2e-5, 5e-5, 1e-6], default=1e-6)
    args = p.parse_args()
    main(args)
