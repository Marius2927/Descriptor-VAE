import argparse
import torch
import torch.nn.functional as F
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split
from CliffordGreen import GaussianPointDescriptor
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from model import VAE, CliffordGreenDescriptorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
Z_DIM = 4
HIDDEN_SIZE = 64
LEARNING_RATE = 1e-6
NUM_EPOCHS = 10000

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

def compute_recon_mse(vae, loader):
    total = 0.0
    with torch.no_grad():
        for D in loader:
            x      = D.squeeze(0).unsqueeze(0)
            mu_q, _= vae.encoder(x)
            z      = mu_q.detach()
            out    = vae.decoder(z, 1)
            mu_pred, _ = out.chunk(2, dim=-1)
            total += F.mse_loss(mu_pred, x, reduction="sum").item()
    return total / len(loader.dataset)


def kabsch_rmsd(A: torch.Tensor, B: torch.Tensor) -> float:
    """
    Compute RMSD between two point clouds A, B of shape [N,3] after optimal alignment.
    """
    # center
    A0 = A - A.mean(dim=0, keepdim=True)
    B0 = B - B.mean(dim=0, keepdim=True)
    # covariance
    C = A0.T @ B0  # [3,3
    U, S, Vt = torch.svd(C)
    d = torch.det(Vt @ U.T)
    D = torch.diag(torch.tensor([1.0, 1.0, d], device=A.device))
    R = Vt @ D @ U.T
    A_rot = (A0 @ R.T)
    return torch.sqrt( ((A_rot - B0)**2).mean() ).item()

def descriptor_mse(D_true: torch.Tensor, D_hat: torch.Tensor) -> float:
    """
    MSE over a single descriptor vector
    """
    return torch.mean( (D_true - D_hat)**2 ).item()

def main(args):
    Z_DIM = args.z_dim
    HIDDEN_SIZE = args.hidden_size
    NUM_EPOCHS = args.num_epochs
    pyro.clear_param_store()

    pdb_files = ["data/1unc.pdb", "data/1fsd.pdb"]
    dataset = CliffordGreenDescriptorDataset(pdb_files)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    descriptor_dim = dataset.D.shape[1]
    D_mean = dataset.D.mean(dim=0)
    D_std = dataset.D.std(dim=0) + 1e-6

    vae = VAE(descriptor_dim, z_dim=Z_DIM, hidden_size=HIDDEN_SIZE, D_mean=D_mean, D_std=D_std)
    optimizer = Adam({"lr": LEARNING_RATE})
    svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

    train_elbo = []
    test_elbo = []
    train_mse = []
    train_rmsd = []
    latent_hist = []

    for epoch in range(NUM_EPOCHS):
        beta = min(1.0, epoch / 100)
        avg_elbo = train(svi, train_loader, beta)
        train_elbo.append(-avg_elbo)
        #print("Epoch:", epoch, " ELBO:",avg_elbo)

        with torch.no_grad():
            mse_accum = 0.0
            rmsd_accum = 0.0
            n_models = 0
            z_vals = []


            for D_norm, X_true in train_loader:
                B, Lmax = D_norm.shape
                mu, sigma = vae.encoder(D_norm.to(device))
                z = mu  # [B, z_dim]
                z_vals.append(z.cpu())

                all_params = vae.decoder(z)
                half = all_params.size(1) // 2
                mu_norm = all_params[:, :half]
                D_rec_raw = mu_norm * D_std + D_mean
                D_true_raw = D_norm * D_std + D_mean

                mse_accum += descriptor_mse(D_true_raw, D_rec_raw)

                for b in range(B):
                    X_rec = GaussianPointDescriptor.descriptor_to_coordinates(D_rec_raw[b])
                    rmsd_accum += kabsch_rmsd(X_true[b], X_rec)
                    n_models += 1
            train_mse.append(mse_accum / n_models)
            train_rmsd.append(rmsd_accum / n_models)
            print("Epoch:", epoch, " ELBO:", avg_elbo, "RMSD:", rmsd_accum)

            z_all = torch.cat(z_vals, dim=0).view(-1).numpy()
            latent_hist.append(z_all)

    torch.save({
        "desc_dim": descriptor_dim,
        "D_mean": D_mean.cpu(),
        "D_std": D_std.cpu(),
        "state_dict": vae.state_dict()
    }, "vae_checkpoint.pt")

    # Loss curves
    plt.figure()
    plt.plot(train_elbo, label="Train ELBO")
    plt.xlabel("Epoch")
    plt.ylabel("ELBO")
    plt.legend()
    plt.savefig("plt10.png")

    # MSE & RMSD
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_mse, label="Desc-MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_rmsd, label="3D-RMSD")
    plt.xlabel("Epoch")
    plt.ylabel("RMSD (Å)")
    plt.legend()
    plt.savefig("plt2.png")

    D_np = dataset.D.numpy()
    pca = PCA(n_components=2)
    D_2d = pca.fit_transform(D_np)

    plt.figure(figsize=(6, 5))
    plt.scatter(D_2d[:, 0], D_2d[:, 1], alpha=0.7)
    plt.title("PCA of Clifford-Green Descriptors")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.savefig("plt3.png")

    pca_3d = PCA(n_components=3)
    D_3d = pca_3d.fit_transform(D_np)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(D_3d[:, 0], D_3d[:, 1], D_3d[:, 2], alpha=0.7)
    ax.set_title("3D PCA of Descriptors")
    plt.savefig("plt4.png")

    for i, z_all in enumerate(latent_hist[::max(1, NUM_EPOCHS // 5)]):  # 5 snapshots
        plt.figure()
        plt.hist(z_all, bins=50, density=True)
        plt.title(f"Epoch {i * (NUM_EPOCHS // 5)} latent z distribution")
        plt.xlabel("z")
        plt.ylabel("Density")
        plt.savefig("plt"+str(i+5)+".png")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--z_dim", choices=[4, 8, 16, 32], default=4)
    p.add_argument("--hidden_size", choices=[32, 64, 128, 256], default=64)
    p.add_argument("--num_epochs", choices=[1000, 5000, 10000], default=500)
    args = p.parse_args()
    main(args)
