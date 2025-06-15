import torch
ATOL = 0.001

class GaussianPointDescriptor:
    @staticmethod
    def coordinates_to_descriptor(X: torch.Tensor) -> torch.Tensor:
        """
        Descriptor of length 3*(N-1):
        For each i=1,...,N-1, store V^T * (X_{i+1} - mean(X[:i])) in 3 coords.
        """
        N, d = X.shape
        assert d == 3, "Coordinates must be 3D."
        # compute running cumulative sum and then running mean:
        X_cumsum = torch.cumsum(X, dim=0)   # [N,3]
        counts = torch.arange(1, N+1, device=X.device).view(-1,1)
        X_mean = X_cumsum / counts    # [N,3]

        # compute displacements D_i = X[i+1] − X_mean[i] for i=0..N-2
        diffs = X[1:] - X_mean[:-1]

        # Build local orthonormal basis V = [v1, v2, v3] via Gram–Schmidt
        V = torch.zeros((3,3), device=X.device, dtype=X.dtype)
        for k in range(3):
            v = diffs[k].clone()
            # subtract projections onto previous basis vectors
            for j in range(k):
                v = v - (diffs[k] @ V[j]) * V[j]
            norm = v.norm()
            V[k] = v / norm if norm>1e-6 else torch.zeros_like(v)

        # For each D_i, compute its coordinates in basis V
        D = torch.zeros((N-1)*3, device=X.device, dtype=X.dtype)
        for i in range(N-1):
            proj = diffs[i] @ V.T
            D[3*i : 3*i+3] = proj

        return D    # [3*(N-1)]


    @staticmethod
    def descriptor_to_coordinates(D: torch.Tensor) -> torch.Tensor:
        """
        Invert the descriptor of length 3*(N-1) back to [N,3] coords.
        """
        L = D.shape[0]
        assert L % 3 == 0, "Descriptor length must be multiple of 3"
        N = L//3 + 1
        device, dtype = D.device, D.dtype

        # Interpret D as N-1 local-coordinate vectors
        fake_diffs = D.view(N-1, 3)  # [N-1,3]

        # Recompute basis V by Gram–Schmidt on the first 3 fake diffs
        V = torch.zeros((3,3), device=device, dtype=dtype)
        for k in range(3):
            v = fake_diffs[k].clone()
            for j in range(k):
                v = v - (fake_diffs[k] @ V[j]) * V[j]
            norm = v.norm()
            V[k] = v / norm if norm>1e-6 else torch.zeros_like(v)

        # Transform all local coords back into real-space displacements
        diffs = fake_diffs @ V       # [N-1,3]

        # Sequentially rebuild X using running means + diffs
        X = torch.zeros((N,3), device=device, dtype=dtype)
        for i in range(N-1):
            prev_mean = X[:i+1].mean(dim=0)
            X[i+1] = prev_mean + diffs[i]

        # Recenter the result
        X = X - X.mean(dim=0)
        return X


if __name__ == "__main__":
    # Example test
    X = torch.randn(8, 10, 3)  # Batch of 8 protein structures, 10 atoms each, 3D
    D = GaussianPointDescriptor.coordinates_to_descriptor(X)
    print("Batched descriptor shape:", D.shape)
