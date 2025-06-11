import torch
ATOL = 0.001

class GaussianPointDescriptor:
    @staticmethod
    def coordinates_to_descriptor(X: torch.Tensor) -> torch.Tensor:
        """
        Full descriptor of length 3*(N-1):
        For each i=1..N-1, store V^T * (X_{i+1} - mean(X[:i])) in 3 coords.
        """
        N, d = X.shape
        assert d == 3, "Coordinates must be 3D."

        X_cumsum = torch.cumsum(X, dim=0)                     # [N,3]
        counts   = torch.arange(1, N+1, device=X.device).view(-1,1)
        X_mean   = X_cumsum / counts                          # running means
        diffs    = X[1:] - X_mean[:-1]                        # [N-1,3]

        V = torch.zeros((3,3), device=X.device, dtype=X.dtype)
        for k in range(3):
            v = diffs[k].clone()
            for j in range(k):
                v = v - (diffs[k] @ V[j]) * V[j]
            norm = v.norm()
            V[k] = v / norm if norm>1e-6 else torch.zeros_like(v)

        D = torch.zeros((N-1)*3, device=X.device, dtype=X.dtype)
        for i in range(N-1):
            proj = diffs[i] @ V.T    # [3]
            D[3*i : 3*i+3] = proj

        return D


    @staticmethod
    def descriptor_to_coordinates(D: torch.Tensor) -> torch.Tensor:
        """
        Invert the full descriptor of length 3*(N-1) back to [N,3] coords.
        """
        L = D.shape[0]
        assert L % 3 == 0, "Descriptor length must be multiple of 3"
        N = L//3 + 1
        device, dtype = D.device, D.dtype

        fake_diffs = D.view(N-1, 3)  # [N-1,3]

        V = torch.zeros((3,3), device=device, dtype=dtype)
        for k in range(3):
            v = fake_diffs[k].clone()
            for j in range(k):
                v = v - (fake_diffs[k] @ V[j]) * V[j]
            norm = v.norm()
            V[k] = v / norm if norm>1e-6 else torch.zeros_like(v)

        diffs = fake_diffs @ V       # [N-1,3]

        X = torch.zeros((N,3), device=device, dtype=dtype)
        for i in range(N-1):
            prev_mean = X[:i+1].mean(dim=0)
            X[i+1] = prev_mean + diffs[i]

        X = X - X.mean(dim=0)
        return X


if __name__ == "__main__":
    # Example test
    X = torch.randn(8, 10, 3)  # Batch of 8 protein structures, 10 atoms each, 3D
    D = GaussianPointDescriptor.coordinates_to_descriptor(X)
    print("Batched descriptor shape:", D.shape)
