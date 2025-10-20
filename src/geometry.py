"""
Spatial geometry and graph Laplacian construction.
Supports 2D grids, 1D polylines, and arbitrary road graphs.
"""

import numpy as np
import scipy.sparse as sp
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class Geometry:
    """Container for spatial geometry information."""
    mode: str
    n: int
    coords: np.ndarray  # (n, 2) or (n, 3) coordinates
    adjacency: sp.spmatrix  # Sparse adjacency matrix
    laplacian: sp.spmatrix  # Graph Laplacian
    h: Optional[float] = None  # Grid spacing (for regular grids)

    def get_neighbors(self, idx: int) -> np.ndarray:
        """Get neighbor indices for a given node."""
        row = self.adjacency.getrow(idx)
        return row.indices

    def distance(self, i: int, j: int) -> float:
        """Euclidean distance between nodes i and j."""
        return np.linalg.norm(self.coords[i] - self.coords[j])


def build_grid2d_geometry(nx: int, ny: int, h: float) -> Geometry:
    """
    Build regular 2D grid geometry.

    Args:
        nx, ny: Grid dimensions
        h: Grid spacing (meters)

    Returns:
        Geometry object with 4-connected grid graph
    """
    n = nx * ny

    # Generate coordinates
    x = np.arange(nx) * h
    y = np.arange(ny) * h
    xx, yy = np.meshgrid(x, y, indexing='ij')
    coords = np.column_stack([xx.ravel(), yy.ravel()])

    # Build adjacency matrix (4-connected)
    def idx(i, j):
        """Convert (i,j) grid coordinates to linear index."""
        return i * ny + j

    row_idx = []
    col_idx = []

    for i in range(nx):
        for j in range(ny):
            current = idx(i, j)
            # Right neighbor
            if i < nx - 1:
                row_idx.extend([current, idx(i + 1, j)])
                col_idx.extend([idx(i + 1, j), current])
            # Top neighbor
            if j < ny - 1:
                row_idx.extend([current, idx(i, j + 1)])
                col_idx.extend([idx(i, j + 1), current])

    adjacency = sp.csr_matrix(
        (np.ones(len(row_idx)), (row_idx, col_idx)),
        shape=(n, n)
    )

    # Compute graph Laplacian L = D - A
    degrees = np.array(adjacency.sum(axis=1)).flatten()
    laplacian = sp.diags(degrees) - adjacency

    return Geometry(
        mode="grid2d",
        n=n,
        coords=coords,
        adjacency=adjacency.tocsr(),
        laplacian=laplacian.tocsr(),
        h=h
    )


def build_graph_laplacian(adjacency: sp.spmatrix,
                          normalized: bool = False) -> sp.spmatrix:
    """
    Build graph Laplacian from adjacency matrix.

    Args:
        adjacency: Sparse adjacency matrix (symmetric)
        normalized: If True, compute normalized Laplacian

    Returns:
        Graph Laplacian matrix L
    """
    n = adjacency.shape[0]
    degrees = np.array(adjacency.sum(axis=1)).flatten()

    if not normalized:
        # Combinatorial Laplacian: L = D - A
        L = sp.diags(degrees) - adjacency
    else:
        # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        d_inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        L = sp.eye(n) - D_inv_sqrt @ adjacency @ D_inv_sqrt

    return L.tocsr()


def build_polyline1d_geometry(n: int, segment_length: float) -> Geometry:
    """
    Build 1D polyline geometry (road centerline).

    Args:
        n: Number of segments
        segment_length: Length per segment (meters)

    Returns:
        Geometry object with 1D chain graph
    """
    # Linear coordinates
    coords = np.column_stack([
        np.arange(n) * segment_length,
        np.zeros(n)
    ])

    # Build chain adjacency
    row_idx = list(range(n - 1)) + list(range(1, n))
    col_idx = list(range(1, n)) + list(range(n - 1))
    adjacency = sp.csr_matrix(
        (np.ones(len(row_idx)), (row_idx, col_idx)),
        shape=(n, n)
    )

    laplacian = build_graph_laplacian(adjacency)

    return Geometry(
        mode="polyline1d",
        n=n,
        coords=coords,
        adjacency=adjacency,
        laplacian=laplacian,
        h=segment_length
    )


def compute_distance_matrix(coords: np.ndarray,
                            sparse_threshold: float = None) -> sp.spmatrix:
    """
    Compute pairwise Euclidean distances.

    Args:
        coords: (n, d) coordinate array
        sparse_threshold: If given, zero out distances > threshold

    Returns:
        Distance matrix (sparse if threshold given)
    """
    from scipy.spatial.distance import cdist
    D = cdist(coords, coords, metric='euclidean')

    if sparse_threshold is not None:
        # Sparsify by thresholding
        D[D > sparse_threshold] = 0
        return sp.csr_matrix(D)
    else:
        return D


def get_spatial_blocks(coords: np.ndarray,
                       k: int,
                       strategy: str = "kmeans",
                       rng: np.random.Generator = None) -> np.ndarray:
    """
    Partition spatial domain into k contiguous blocks.

    Args:
        coords: (n, d) spatial coordinates
        k: Number of blocks
        strategy: "kmeans" | "grid" | "random"
        rng: Random number generator

    Returns:
        block_labels: (n,) array of block indices 0..k-1
    """
    n = len(coords)

    if strategy == "kmeans":
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, random_state=rng.integers(0, 2 ** 31), n_init=10)
        block_labels = kmeans.fit_predict(coords)

    elif strategy == "grid":
        # Simple grid-based partitioning
        if coords.shape[1] == 2:
            # 2D: divide into k≈sqrt(k)×sqrt(k) grid
            k_side = int(np.ceil(np.sqrt(k)))
            x_edges = np.linspace(coords[:, 0].min(), coords[:, 0].max(), k_side + 1)
            y_edges = np.linspace(coords[:, 1].min(), coords[:, 1].max(), k_side + 1)

            x_idx = np.digitize(coords[:, 0], x_edges[1:-1])
            y_idx = np.digitize(coords[:, 1], y_edges[1:-1])
            block_labels = x_idx * k_side + y_idx

            # Relabel to 0..k-1
            unique_labels = np.unique(block_labels)
            label_map = {old: new for new, old in enumerate(unique_labels)}
            block_labels = np.array([label_map[l] for l in block_labels])
        else:
            raise NotImplementedError("Grid partitioning only for 2D")

    elif strategy == "random":
        # Random assignment (not spatially contiguous)
        block_labels = rng.integers(0, k, size=n)

    else:
        raise ValueError(f"Unknown block strategy: {strategy}")

    return block_labels


if __name__ == "__main__":
    # Test grid construction
    geom = build_grid2d_geometry(10, 10, h=5.0)
    print(f"Grid: n={geom.n}, nnz(L)={geom.laplacian.nnz}")
    print(f"Laplacian sparsity: {geom.laplacian.nnz / geom.n ** 2 * 100:.2f}%")

    # Test spatial blocking
    rng = np.random.default_rng(42)
    blocks = get_spatial_blocks(geom.coords, k=5, strategy="kmeans", rng=rng)
    print(f"Block sizes: {np.bincount(blocks)}")