"""
LSTGCNN model for traffic prediction at different granularities.
Used as the unified backbone for the lane granularity diagnostic study.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphConvolution(nn.Module):
    """Simple graph convolution layer: H' = sigma(A * H * W + b)"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, adj):
        # x: (batch, num_nodes, features)
        # adj: (num_nodes, num_nodes)
        x = torch.matmul(adj, x)  # aggregate neighbors
        x = self.linear(x)
        return x


class LSTGCNN(nn.Module):
    """
    LSTGCNN: LSTM + Graph Convolution for spatiotemporal traffic prediction.

    Architecture:
    1. Temporal: LSTM captures temporal dependencies
    2. Spatial: Graph convolution captures spatial dependencies
    3. Output: Linear projection to prediction horizon

    Args:
        num_nodes: number of nodes in the graph
        input_dim: input feature dimension (default: 1 for speed)
        hidden_dim: hidden dimension
        output_dim: prediction horizon
        num_gc_layers: number of graph convolution layers
        num_lstm_layers: number of LSTM layers
        dropout: dropout rate
        adj: adjacency matrix (num_nodes x num_nodes), optional
    """
    def __init__(self, num_nodes, input_dim=1, hidden_dim=64, output_dim=3,
                 num_gc_layers=2, num_lstm_layers=2, dropout=0.1, adj=None):
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_gc_layers = num_gc_layers

        # Temporal encoder (LSTM)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
        )

        # Spatial encoder (Graph Convolution layers)
        self.gc_layers = nn.ModuleList()
        self.gc_layers.append(GraphConvolution(hidden_dim, hidden_dim))
        for _ in range(num_gc_layers - 1):
            self.gc_layers.append(GraphConvolution(hidden_dim, hidden_dim))

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Store adjacency
        if adj is not None:
            self.register_buffer("adj", torch.tensor(adj, dtype=torch.float32))
        else:
            self.adj = None

    def forward(self, x, adj=None):
        """
        Args:
            x: (batch, lookback, num_nodes) or (batch, lookback, num_nodes, input_dim)
            adj: (num_nodes, num_nodes) adjacency matrix, optional

        Returns:
            output: (batch, horizon, num_nodes) predicted speeds
        """
        if adj is None:
            adj = self.adj

        batch_size, lookback, n = x.shape[0], x.shape[1], x.shape[2]

        # Reshape: (batch, lookback, num_nodes) -> (batch * num_nodes, lookback, 1)
        x_flat = x.permute(0, 2, 1).reshape(batch_size * n, lookback, 1)

        # Temporal encoding via LSTM
        lstm_out, _ = self.lstm(x_flat)  # (batch*n, lookback, hidden)
        lstm_last = lstm_out[:, -1, :]  # (batch*n, hidden) last timestep

        # Reshape back: (batch, num_nodes, hidden)
        h = lstm_last.reshape(batch_size, n, self.hidden_dim)

        # Spatial encoding via Graph Convolution
        for gc in self.gc_layers:
            h = F.relu(gc(h, adj))
            h = self.dropout(h)

        # Output projection: (batch, num_nodes, horizon)
        output = self.output_proj(h)  # (batch, num_nodes, horizon)

        # Reshape to (batch, horizon, num_nodes)
        output = output.permute(0, 2, 1)

        return output


# ============================================================
# v4: GAT (Graph Attention Network) — pure PyTorch, no PyG
# ============================================================

class GATLayer(nn.Module):
    """
    Single multi-head graph attention layer (Velickovic et al. 2018).

    e_ij = LeakyReLU(a_l^T [W h_i || W h_j])
    alpha_ij = softmax_j(e_ij) masked by adjacency
    h_i' = sigma(sum_j alpha_ij * W h_j)
    """
    def __init__(self, in_dim, out_dim, n_heads=4, dropout=0.1, concat=True):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.concat = concat

        # Linear projection: W in R^(in_dim × out_dim)
        self.W = nn.Linear(in_dim, out_dim * n_heads, bias=False)

        # Attention mechanism: a_l in R^(2 * out_dim)
        self.a_src = nn.Parameter(torch.zeros(n_heads, out_dim))
        self.a_dst = nn.Parameter(torch.zeros(n_heads, out_dim))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        if concat:
            self.output_dim = out_dim * n_heads
        else:
            self.output_dim = out_dim

    def forward(self, x, adj):
        """
        Args:
            x: (batch, num_nodes, in_dim)
            adj: (num_nodes, num_nodes) dense adjacency matrix

        Returns:
            (batch, num_nodes, output_dim)
        """
        B, N, _ = x.shape

        # Project: (B, N, n_heads * out_dim) -> (B, N, n_heads, out_dim)
        h = self.W(x).view(B, N, self.n_heads, self.out_dim)

        # Compute attention scores: e_ij = LeakyReLU(a_src^T h_i + a_dst^T h_j)
        # e_src: (B, N, n_heads) = sum over out_dim of (h * a_src)
        e_src = (h * self.a_src).sum(dim=-1)  # (B, N, n_heads)
        e_dst = (h * self.a_dst).sum(dim=-1)  # (B, N, n_heads)

        # Pairwise: e_ij = e_src_i + e_dst_j → (B, N, N, n_heads)
        e = e_src.unsqueeze(2) + e_dst.unsqueeze(1)  # (B, N, N, n_heads)
        e = self.leaky_relu(e)

        # Mask by adjacency: set non-edges to -inf
        # adj: (N, N) → (1, N, N, 1) for broadcasting
        mask = (adj > 0).unsqueeze(0).unsqueeze(-1)  # (1, N, N, 1)
        e = e.masked_fill(~mask, float('-inf'))

        # Softmax over dim=2 (neighbors)
        alpha = torch.softmax(e, dim=2)  # (B, N, N, n_heads)
        alpha = self.dropout(alpha)

        # Store attention weights for diagnostic extraction
        self._last_attention = alpha.detach()

        # Aggregate: (B, N, N, n_heads) × (B, N, n_heads, out_dim) → (B, N, n_heads, out_dim)
        h_out = torch.einsum('bijh,bjhd->bihd', alpha, h)

        if self.concat:
            # Concatenate heads: (B, N, n_heads * out_dim)
            h_out = h_out.reshape(B, N, -1)
        else:
            # Average heads: (B, N, out_dim)
            h_out = h_out.mean(dim=2)

        return h_out


class GATPredictor(nn.Module):
    """
    LSTM + GAT prediction model. Same interface as LSTGCNN.

    Architecture: LSTM (temporal) → GAT layers (spatial) → Linear (output)
    """
    def __init__(self, num_nodes, input_dim=1, hidden_dim=64, output_dim=3,
                 num_gat_layers=2, num_lstm_layers=2, n_heads=4, dropout=0.1, adj=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # Temporal encoder (LSTM)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
        )

        # Spatial encoder (GAT layers)
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATLayer(hidden_dim, hidden_dim // n_heads, n_heads, dropout, concat=True))
        for _ in range(num_gat_layers - 1):
            gat_in = hidden_dim  # first layer output is hidden_dim (n_heads * hidden_dim//n_heads)
            self.gat_layers.append(GATLayer(gat_in, hidden_dim // n_heads, n_heads, dropout, concat=True))

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        # Store adjacency
        if adj is not None:
            self.register_buffer("adj", torch.tensor(adj, dtype=torch.float32))
        else:
            self.adj = None

    def forward(self, x, adj=None):
        """
        Args:
            x: (batch, lookback, num_nodes)
            adj: (num_nodes, num_nodes) optional

        Returns:
            (batch, horizon, num_nodes)
        """
        if adj is None:
            adj = self.adj

        batch_size, lookback, n = x.shape[0], x.shape[1], x.shape[2]

        # Temporal: (B*N, lookback, 1) → LSTM → (B*N, hidden)
        x_flat = x.permute(0, 2, 1).reshape(batch_size * n, lookback, 1)
        lstm_out, _ = self.lstm(x_flat)
        h = lstm_out[:, -1, :].reshape(batch_size, n, self.hidden_dim)

        # Spatial: GAT layers
        for gat in self.gat_layers:
            h = F.relu(gat(h, adj))
            h = self.dropout(h)

        # Output: (B, N, horizon) → (B, horizon, N)
        output = self.output_proj(h).permute(0, 2, 1)
        return output

    def extract_attention(self, x, adj=None):
        """Extract attention weights from all GAT layers.

        Returns:
            list of (B, N, N, n_heads) tensors, one per GAT layer.
        """
        if adj is None:
            adj = self.adj
        batch_size, lookback, n = x.shape[0], x.shape[1], x.shape[2]

        x_flat = x.permute(0, 2, 1).reshape(batch_size * n, lookback, 1)
        lstm_out, _ = self.lstm(x_flat)
        h = lstm_out[:, -1, :].reshape(batch_size, n, self.hidden_dim)

        attention_weights = []
        for gat in self.gat_layers:
            h_out = gat(h, adj)
            attention_weights.append(gat._last_attention)
            h = F.relu(h_out)
            h = self.dropout(h)

        return attention_weights



class RandomGATPredictor(nn.Module):
    """GAT with fixed (uniform) attention -- equivalent to degree-normalized GCN.

    Same architecture as GATPredictor but attention is fixed to adjacency-based
    degree normalization, isolating the contribution of learned attention.
    """
    def __init__(self, num_nodes, input_dim=1, hidden_dim=64, output_dim=3,
                 num_gat_layers=2, num_lstm_layers=2, n_heads=4, dropout=0.1, adj=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_lstm_layers, batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
        )

        # Reuse GATLayer for the linear projection part
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATLayer(hidden_dim, hidden_dim // n_heads, n_heads, dropout, concat=True))
        for _ in range(num_gat_layers - 1):
            self.gat_layers.append(GATLayer(hidden_dim, hidden_dim // n_heads, n_heads, dropout, concat=True))

        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        if adj is not None:
            self.register_buffer("adj", torch.tensor(adj, dtype=torch.float32))
        else:
            self.adj = None

    def forward(self, x, adj=None):
        if adj is None:
            adj = self.adj
        batch_size, lookback, n = x.shape[0], x.shape[1], x.shape[2]

        x_flat = x.permute(0, 2, 1).reshape(batch_size * n, lookback, 1)
        lstm_out, _ = self.lstm(x_flat)
        h = lstm_out[:, -1, :].reshape(batch_size, n, self.hidden_dim)

        # Fixed attention: degree-normalized adjacency
        adj_norm = adj.clone()
        deg = adj_norm.sum(dim=1, keepdim=True)
        deg[deg == 0] = 1.0
        adj_norm = adj_norm / deg  # (N, N)

        for gat in self.gat_layers:
            # Use GATLayer's linear projection but skip its attention
            h_proj = gat.W(h).view(batch_size, n, gat.n_heads, gat.out_dim)
            # Fixed attention: adj_norm broadcast to (B, N, N, n_heads)
            alpha_fixed = adj_norm.unsqueeze(0).unsqueeze(-1).expand(
                batch_size, n, n, gat.n_heads)
            h_out = torch.einsum('bijh,bjhd->bihd', alpha_fixed, h_proj)
            if gat.concat:
                h_out = h_out.reshape(batch_size, n, -1)
            else:
                h_out = h_out.mean(dim=2)
            h = F.relu(h_out)
            h = self.dropout(h)

        output = self.output_proj(h).permute(0, 2, 1)
        return output


class PureLSTMPredictor(nn.Module):
    """
    Pure LSTM with no spatial message passing. Spatial ablation baseline.

    Each node processed independently with shared LSTM weights.
    """
    def __init__(self, num_nodes, input_dim=1, hidden_dim=64, output_dim=3,
                 num_lstm_layers=2, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj=None):
        """
        Args:
            x: (batch, lookback, num_nodes) — adj is ignored

        Returns:
            (batch, horizon, num_nodes)
        """
        batch_size, lookback, n = x.shape[0], x.shape[1], x.shape[2]

        # Reshape: (B, T, N) → (B*N, T, 1)
        x_flat = x.permute(0, 2, 1).reshape(batch_size * n, lookback, 1)

        # LSTM
        lstm_out, _ = self.lstm(x_flat)
        h = lstm_out[:, -1, :]  # (B*N, hidden)
        h = self.dropout(h)

        # Output: (B*N, horizon) → (B, horizon, N)
        output = self.output_proj(h)  # (B*N, horizon)
        output = output.reshape(batch_size, n, -1).permute(0, 2, 1)
        return output


def train_model_with_diagnostics(model, X_train, Y_train, X_val, Y_val, adj,
                                  lr=0.001, max_epochs=100, patience=15,
                                  device="cuda", batch_size=256):
    """
    Extended training with per-epoch gradient norm logging (for S2 diagnostics).

    Returns:
        model: best model
        history: dict with train_losses, val_losses, grad_norms
    """
    model = model.to(device)
    adj_tensor = torch.tensor(adj, dtype=torch.float32).to(device)

    # Pre-load all data to GPU to avoid per-batch CPU→GPU transfers
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    n_train = len(X_train_t)
    n_val = len(X_val_t)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "grad_norm": []}

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0
        total_grad_norm = 0.0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_train_t[idx]
            yb = Y_train_t[idx]

            optimizer.zero_grad()
            pred = model(xb, adj_tensor)
            loss = criterion(pred, yb)
            loss.backward()

            # Compute gradient norm before step
            grad_sq_sum = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_sq_sum += p.grad.data.norm(2).item() ** 2
            total_grad_norm += grad_sq_sum ** 0.5

            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        history["train_loss"].append(epoch_loss / n_batches)
        history["grad_norm"].append(total_grad_norm / max(n_batches, 1))

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for j in range(0, n_val, batch_size):
                xv = X_val_t[j:j + batch_size]
                yv = Y_val_t[j:j + batch_size]
                val_pred = model(xv, adj_tensor)
                val_loss += criterion(val_pred, yv).item()
                val_batches += 1
        val_loss /= val_batches
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch + 1}: train={history['train_loss'][-1]:.4f}, "
                  f"val={val_loss:.4f}, grad_norm={history['grad_norm'][-1]:.4f}")

    model.load_state_dict(best_state)
    model = model.cpu()
    return model, history


# ============================================================
# STGCN (Yu et al., IJCAI 2018)
# Chebyshev graph convolution (K=3) + 1D causal temporal convolution
# ============================================================

class ChebConv(nn.Module):
    """Chebyshev spectral graph convolution: T_k(L_norm) = 2*L_norm*T_{k-1} - T_{k-2}"""
    def __init__(self, in_dim, out_dim, K=3):
        super().__init__()
        self.K = K
        self.weights = nn.Parameter(torch.FloatTensor(K, in_dim, out_dim))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x, laplacian):
        """
        Args:
            x: (batch, num_nodes, in_dim)
            laplacian: (num_nodes, num_nodes) normalized Laplacian L = I - D^{-1/2}AD^{-1/2}
        Returns:
            (batch, num_nodes, out_dim)
        """
        B, N, C = x.shape

        # Chebyshev polynomials: T_0 = I, T_1 = L_norm
        T = [x]  # T_0(x) = x
        if self.K > 1:
            T.append(torch.matmul(laplacian, x))  # T_1(x) = L * x

        for k in range(2, self.K):
            # T_k = 2 * L * T_{k-1} - T_{k-2}
            T_k = 2 * torch.matmul(laplacian, T[-1]) - T[-2]
            T.append(T_k)

        # Weighted sum: sum_k T_k * W_k
        out = torch.zeros(B, N, self.weights.shape[-1], device=x.device)
        for k in range(self.K):
            out += torch.matmul(T[k], self.weights[k])

        return out


class TemporalConv(nn.Module):
    """1D causal convolution on temporal axis (no future leakage)."""
    def __init__(self, in_dim, out_dim, kernel_size=3):
        super().__init__()
        # Causal padding: pad left (kernel_size - 1)
        self.padding = kernel_size - 1
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size, 1))
        # Gated activation
        self.conv_gate = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size, 1))

    def forward(self, x):
        """
        Args:
            x: (batch, channels, time, nodes)
        Returns:
            (batch, channels, time, nodes)
        """
        # Pad on temporal axis (left side only for causality)
        x_padded = F.pad(x, (0, 0, self.padding, 0))

        h = torch.tanh(self.conv(x_padded))
        g = torch.sigmoid(self.conv_gate(x_padded))
        return h * g  # gated activation


class STGCNBlock(nn.Module):
    """One STGCN block: Temporal → Spatial → Temporal."""
    def __init__(self, channels, K=3):
        super().__init__()
        self.temporal1 = TemporalConv(channels, channels, kernel_size=3)
        self.chebconv = ChebConv(channels, channels, K=K)
        self.temporal2 = TemporalConv(channels, channels, kernel_size=3)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x, laplacian):
        """
        Args:
            x: (batch, channels, time, nodes)
            laplacian: (nodes, nodes)
        Returns:
            (batch, channels, time, nodes)
        """
        # Temporal conv
        t = self.temporal1(x)

        # Spatial Chebyshev conv: (B,C,T,N) → (B*T, N, C) → ChebConv → (B,C,T,N)
        B, C, T, N = t.shape
        t_reshaped = t.permute(0, 2, 1, 3).reshape(B * T, N, C)
        s = self.chebconv(t_reshaped, laplacian)
        s = s.reshape(B, T, N, C).permute(0, 3, 1, 2)

        # Temporal conv
        t2 = self.temporal2(s)

        # Residual + BN
        return self.bn(t2 + x)


class STGCNPredictor(nn.Module):
    """
    STGCN (Yu et al., IJCAI 2018) for traffic prediction.

    Same interface as LSTGCNN/GAT: forward(x, adj) → (batch, horizon, num_nodes).
    Uses Chebyshev spectral graph convolution (K=3) + 1D causal temporal convolution.
    """
    def __init__(self, num_nodes, input_dim=1, hidden_dim=64, output_dim=3,
                 num_blocks=2, K=3, dropout=0.1, adj=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # Input projection: 1 → hidden_dim
        self.input_proj = nn.Conv2d(input_dim, hidden_dim, kernel_size=(1, 1))

        # STGCN blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(STGCNBlock(hidden_dim, K=K))

        # Output: global average pool on time axis → project to horizon
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        # Store adjacency and compute normalized Laplacian
        if adj is not None:
            self._init_graph(adj)
        else:
            self.laplacian = None

    def _init_graph(self, adj):
        """Compute normalized Laplacian L = I - D^{-1/2} A D^{-1/2}."""
        adj_t = torch.tensor(adj, dtype=torch.float32)
        N = adj_t.shape[0]
        # Add self-loops
        adj_t = adj_t + torch.eye(N)
        # Degree
        deg = adj_t.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        # Normalized: D^{-1/2} A D^{-1/2}
        adj_norm = D_inv_sqrt @ adj_t @ D_inv_sqrt
        # Laplacian: L = I - A_norm
        laplacian = torch.eye(N) - adj_norm
        self.register_buffer("laplacian", laplacian)
        self.register_buffer("adj_norm", adj_norm)

    def forward(self, x, adj=None):
        """
        Args:
            x: (batch, lookback, num_nodes) or (batch, lookback, num_nodes, 1)
            adj: unused (use stored laplacian)
        Returns:
            (batch, horizon, num_nodes)
        """
        B, T, N = x.shape[0], x.shape[1], x.shape[2]

        # Reshape to (B, 1, T, N) — (batch, channels, time, nodes)
        x = x.unsqueeze(1)
        if x.dim() == 5:
            x = x.squeeze(-1).unsqueeze(1)

        # Input projection
        h = self.input_proj(x)  # (B, hidden, T, N)
        h = self.dropout(h)

        # STGCN blocks
        for block in self.blocks:
            h = block(h, self.laplacian)
            h = self.dropout(h)

        # Global average pooling on time axis
        h = h.mean(dim=2)  # (B, hidden, N)
        h = h.permute(0, 2, 1)  # (B, N, hidden)

        # Output: (B, N, horizon) → (B, horizon, N)
        out = self.output_proj(h).permute(0, 2, 1)
        return out


# ============================================================
# DCRNN (Li et al., ICLR 2018)
# Diffusion convolution + encoder-decoder GRU
# ============================================================

class DiffusionConv(nn.Module):
    """
    Diffusion convolution: X * G = sum_{k=0}^{K-1} (D_o^{-1} A)^k X W_k

    K=2 steps: forward + backward diffusion.
    """
    def __init__(self, in_dim, out_dim, K=2):
        super().__init__()
        self.K = K
        self.weights = nn.Parameter(torch.FloatTensor(K, in_dim, out_dim))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x, adj_fwd, adj_bwd):
        """
        Args:
            x: (batch, num_nodes, in_dim)
            adj_fwd: (N, N) D_o^{-1} A (row-normalized forward)
            adj_bwd: (N, N) D_i^{-1} A^T (column-normalized backward)
        Returns:
            (batch, num_nodes, out_dim)
        """
        B, N, C = x.shape
        out = torch.zeros(B, N, self.weights.shape[-1], device=x.device)

        # k=0: self-loop (identity)
        out += torch.matmul(x, self.weights[0])

        # k=1..K-1: diffusion steps (alternate forward and backward)
        h_fwd = x
        h_bwd = x
        for k in range(1, self.K):
            if k % 2 == 1:
                h_fwd = torch.matmul(adj_fwd, h_fwd)
                out += torch.matmul(h_fwd, self.weights[k])
            else:
                h_bwd = torch.matmul(adj_bwd, h_bwd)
                out += torch.matmul(h_bwd, self.weights[k])

        return out


class DCGRUCell(nn.Module):
    """Diffusion Convolutional GRU cell."""
    def __init__(self, input_dim, hidden_dim, K=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate_conv = DiffusionConv(input_dim + hidden_dim, 2 * hidden_dim, K=K)
        self.candidate_conv = DiffusionConv(input_dim + hidden_dim, hidden_dim, K=K)

    def forward(self, x, h, adj_fwd, adj_bwd):
        """
        Args:
            x: (batch, nodes, input_dim)
            h: (batch, nodes, hidden_dim)
        Returns:
            h_new: (batch, nodes, hidden_dim)
        """
        combined = torch.cat([x, h], dim=-1)  # (B, N, input_dim + hidden_dim)

        gates = self.gate_conv(combined, adj_fwd, adj_bwd)
        r, z = torch.split(gates, self.hidden_dim, dim=-1)
        r = torch.sigmoid(r)
        z = torch.sigmoid(z)

        candidate = torch.cat([x, r * h], dim=-1)
        h_tilde = torch.tanh(self.candidate_conv(candidate, adj_fwd, adj_bwd))

        h_new = (1 - z) * h + z * h_tilde
        return h_new


class DCRNNPredictor(nn.Module):
    """
    DCRNN (Li et al., ICLR 2018) for traffic prediction.

    Encoder-decoder with Diffusion Convolutional GRU.
    Same interface: forward(x, adj) → (batch, horizon, num_nodes).
    """
    def __init__(self, num_nodes, input_dim=1, hidden_dim=64, output_dim=3,
                 num_rnn_layers=2, K=2, dropout=0.1, adj=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_rnn_layers = num_rnn_layers
        self.output_dim = output_dim  # horizon

        # Encoder layers
        self.encoder_cells = nn.ModuleList()
        self.encoder_cells.append(DCGRUCell(input_dim, hidden_dim, K=K))
        for _ in range(num_rnn_layers - 1):
            self.encoder_cells.append(DCGRUCell(hidden_dim, hidden_dim, K=K))

        # Decoder layers (with teacher forcing: input = hidden_dim)
        self.decoder_cells = nn.ModuleList()
        for _ in range(num_rnn_layers):
            self.decoder_cells.append(DCGRUCell(hidden_dim, hidden_dim, K=K))

        # Output projection per decoder step
        self.output_proj = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)

        # Graph matrices
        if adj is not None:
            self._init_graph(adj)
        else:
            self.adj_fwd = None
            self.adj_bwd = None

    def _init_graph(self, adj):
        """Compute forward and backward diffusion matrices."""
        adj_t = torch.tensor(adj, dtype=torch.float32)
        N = adj_t.shape[0]

        # Forward: row-normalized D_o^{-1} A
        row_sum = adj_t.sum(dim=1, keepdim=True)
        row_sum[row_sum == 0] = 1.0
        adj_fwd = adj_t / row_sum

        # Backward: column-normalized D_i^{-1} A^T
        col_sum = adj_t.sum(dim=0, keepdim=True)
        col_sum[col_sum == 0] = 1.0
        adj_bwd = adj_t.T / col_sum

        self.register_buffer("adj_fwd", adj_fwd)
        self.register_buffer("adj_bwd", adj_bwd)

    def forward(self, x, adj=None):
        """
        Args:
            x: (batch, lookback, num_nodes)
            adj: unused (use stored matrices)
        Returns:
            (batch, horizon, num_nodes)
        """
        B, T, N = x.shape[0], x.shape[1], x.shape[2]
        H = self.hidden_dim

        # Reshape: (B, T, N) → (B, N, 1) per timestep
        # Encoder: process lookback timesteps
        h_encoder = [torch.zeros(B, N, H, device=x.device) for _ in range(self.num_rnn_layers)]

        for t in range(T):
            x_t = x[:, t, :].unsqueeze(-1)  # (B, N, 1)
            input_t = x_t

            for layer_idx, cell in enumerate(self.encoder_cells):
                h_encoder[layer_idx] = cell(input_t, h_encoder[layer_idx],
                                             self.adj_fwd, self.adj_bwd)
                h_encoder[layer_idx] = self.dropout(h_encoder[layer_idx])
                input_t = h_encoder[layer_idx]

        # Decoder: generate horizon steps (no teacher forcing)
        h_decoder = [h_encoder[i].clone() for i in range(self.num_rnn_layers)]
        outputs = []

        for step in range(self.output_dim):
            # Decoder input: previous hidden state (zero input for autoregressive)
            input_t = torch.zeros(B, N, H, device=x.device)

            for layer_idx, cell in enumerate(self.decoder_cells):
                h_decoder[layer_idx] = cell(input_t, h_decoder[layer_idx],
                                             self.adj_fwd, self.adj_bwd)
                h_decoder[layer_idx] = self.dropout(h_decoder[layer_idx])
                input_t = h_decoder[layer_idx]

            # Output projection: (B, N, H) → (B, N, 1)
            out_t = self.output_proj(input_t).squeeze(-1)  # (B, N)
            outputs.append(out_t)

        # Stack: (horizon, B, N) → (B, horizon, N)
        output = torch.stack(outputs, dim=1)
        return output


class BaselineModels:
    """Simple baselines for comparison."""

    @staticmethod
    def mean_predictor(X, horizon=3):
        """Predict the mean of input sequence as future values."""
        # X: (batch, lookback, num_nodes)
        mean_val = X.mean(axis=1, keepdims=True)  # (batch, 1, num_nodes)
        return np.repeat(mean_val, horizon, axis=1)  # (batch, horizon, num_nodes)

    @staticmethod
    def last_value_predictor(X, horizon=3):
        """Repeat the last observed value."""
        last_val = X[:, -1:, :]  # (batch, 1, num_nodes)
        return np.repeat(last_val, horizon, axis=1)

    @staticmethod
    def linear_trend_predictor(X, horizon=3):
        """Extrapolate linear trend from last few timesteps."""
        batch, lookback, n = X.shape
        output = np.zeros((batch, horizon, n))

        # Fit linear trend on last 6 timesteps
        t = np.arange(lookback)
        for i in range(batch):
            for j in range(n):
                slope, intercept = np.polyfit(t, X[i, :, j], 1)
                for h in range(horizon):
                    output[i, h, j] = slope * (lookback + h) + intercept

        # Clip to reasonable speed range
        output = np.clip(output, 0, 80)
        return output


def compute_metrics(pred, true, mask=None, baseline_pred=None):
    """
    Compute evaluation metrics (v2 - extended).

    Args:
        pred: (batch, horizon, num_nodes) predictions
        true: (batch, horizon, num_nodes) ground truth
        mask: optional boolean mask for subset of nodes
        baseline_pred: optional (batch, horizon, num_nodes) baseline predictions
                       for skill score computation (e.g., last-value predictor)

    Returns:
        dict of metrics including MAE, RMSE, MAPE, NMAE, skill_score, per_sample_mae
    """
    if mask is not None:
        pred = pred[:, :, mask]
        true = true[:, :, mask]
        if baseline_pred is not None:
            baseline_pred = baseline_pred[:, :, mask]

    mae = np.mean(np.abs(pred - true))
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    mape = np.mean(np.abs(pred - true) / (np.abs(true) + 1e-6)) * 100

    # NMAE: normalized MAE by data range
    y_range = true.max() - true.min()
    nmae = mae / (y_range + 1e-6)

    # Skill score: 1 - MAE_model / MAE_persistence
    # If baseline_pred provided, use it; else compute last-value persistence
    if baseline_pred is not None:
        mae_baseline = np.mean(np.abs(baseline_pred - true))
    else:
        # Default: last-value persistence baseline
        # true shape is (batch, horizon, nodes); persistence = repeat last input value
        # We can't compute this without X_test, so set skill_score=None
        mae_baseline = None

    skill_score = (1 - mae / mae_baseline) if mae_baseline is not None and mae_baseline > 0 else None

    # Per-sample MAE (for statistical tests): average over horizon and nodes
    per_sample_mae = np.mean(np.abs(pred - true), axis=(1, 2))  # (batch,)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "NMAE": nmae,
        "skill_score": skill_score,
        "per_sample_mae": per_sample_mae,
    }


def train_model(model, X_train, Y_train, X_val, Y_val, adj,
                lr=0.001, max_epochs=100, patience=15, device="cuda", batch_size=256):
    """
    Train the model with early stopping (batch training).

    Returns:
        model: best model (by val loss)
        train_losses: list of training losses
        val_losses: list of validation losses
    """
    model = model.to(device)
    adj_tensor = torch.tensor(adj, dtype=torch.float32).to(device)

    # Pre-load all data to GPU to avoid per-batch CPU→GPU transfers
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    n_train = len(X_train_t)
    n_val = len(X_val_t)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_train_t[idx]
            yb = Y_train_t[idx]

            optimizer.zero_grad()
            pred = model(xb, adj_tensor)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        train_losses.append(epoch_loss / n_batches)

        # Validation (batch)
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for j in range(0, n_val, batch_size):
                xv = X_val_t[j:j + batch_size]
                yv = Y_val_t[j:j + batch_size]
                val_pred = model(xv, adj_tensor)
                val_loss += criterion(val_pred, yv).item()
                val_batches += 1
        val_loss /= val_batches
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch + 1}: train={train_losses[-1]:.4f}, val={val_loss:.4f}")

    # Load best model
    model.load_state_dict(best_state)
    model = model.cpu()

    return model, train_losses, val_losses


def evaluate_model(model, X_test, Y_test, adj, device="cpu", scene_mask=None, batch_size=64, scaler=None):
    """
    Evaluate model on test set (batch evaluation).

    v2: Computes skill_score using last-value persistence baseline.

    Returns:
        metrics: dict of overall metrics
        scene_metrics: dict of per-scene metrics
    """
    model = model.to(device)
    model.eval()

    adj_tensor = torch.tensor(adj, dtype=torch.float32).to(device)
    n_test = len(X_test)

    preds = []
    with torch.no_grad():
        for i in range(0, n_test, batch_size):
            xb = torch.tensor(X_test[i:i + batch_size], dtype=torch.float32).to(device)
            pred = model(xb, adj_tensor)
            preds.append(pred.cpu().numpy())

    pred_np = np.concatenate(preds, axis=0)
    true_np = Y_test

    # Last-value persistence baseline (compute in original space for correct skill score)
    horizon = Y_test.shape[1]
    baseline_pred = BaselineModels.last_value_predictor(X_test, horizon=horizon)

    # De-normalize if scaler provided (convert back to original mph space)
    if scaler is not None:
        mean = np.array(scaler['mean']).reshape(1, 1, -1)
        std = np.array(scaler['std']).reshape(1, 1, -1)
        pred_np = pred_np * std + mean
        true_np = true_np * std + mean
        baseline_pred = baseline_pred * std + mean

    # Overall metrics
    metrics = compute_metrics(pred_np, true_np, baseline_pred=baseline_pred)

    # Per-scene metrics
    scene_metrics = {}
    if scene_mask is not None:
        unique_scenes = sorted(set(scene_mask))
        for scene in unique_scenes:
            mask = [i for i, s in enumerate(scene_mask) if s == scene]
            if mask:
                scene_metrics[scene] = compute_metrics(pred_np, true_np, mask,
                                                       baseline_pred=baseline_pred[:, :, mask])

    return metrics, scene_metrics

# ============================================================
# Graph WaveNet (Wu et al., IJCAI 2019)
# ============================================================

class GraphWaveNetPredictor(nn.Module):
    """
    Graph WaveNet adapted for traffic prediction.

    Key innovations:
    1. Self-adaptive adjacency matrix (learnable node embeddings)
    2. Dilated causal temporal convolutions with gating
     3. Spatial graph convolution via matrix multiplication

    Interface: forward(x, adj) -> (batch, horizon, num_nodes)
    """
    def __init__(self, num_nodes, input_dim=1, hidden_dim=64, skip_channels=64,
                 output_dim=3, num_layers=6, dropout=0.3, adj=None,
                 addaptadj=True, aptdim=10):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Temporal dilated conv layers (causal, gated)
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()

        for i in range(num_layers):
            dilation = 2 ** i
            self.filter_convs.append(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=dilation, padding=0))
            self.gate_convs.append(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=dilation, padding=0))
            self.skip_convs.append(nn.Linear(hidden_dim, skip_channels))
            if i < num_layers - 1:
                self.residual_convs.append(nn.Linear(hidden_dim, hidden_dim))

        # Skip aggregation -> output
        self.skip_agg = nn.Linear(skip_channels, skip_channels)
        self.end_conv_1 = nn.Linear(skip_channels, hidden_dim)
        self.end_conv_2 = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        # Self-adaptive adjacency
        if addaptadj:
            self.nodevec1 = nn.Parameter(torch.randn(num_nodes, aptdim))
            self.nodevec2 = nn.Parameter(torch.randn(num_nodes, aptdim))

        # Predefined adjacency
        if adj is not None:
            self.register_buffer("adj_predefined",
                                 torch.tensor(adj, dtype=torch.float32))
            adj_t = torch.tensor(adj, dtype=torch.float32) + torch.eye(num_nodes)
            row_sum = adj_t.sum(dim=1, keepdim=True)
            row_sum[row_sum == 0] = 1.0
            self.register_buffer("adj_norm", adj_t / row_sum)
        else:
            self.adj_predefined = None
            self.adj_norm = None

        self.addaptadj = addaptadj

    def _get_adj(self, adj_param, device):
        """Build total adjacency: predefined (normalized) + adaptive."""
        if adj_param is not None:
            adj_t = adj_param + torch.eye(adj_param.shape[0], device=device)
            row_sum = adj_t.sum(dim=1, keepdim=True)
            row_sum[row_sum == 0] = 1.0
            A = adj_t / row_sum
        elif self.adj_norm is not None:
            A = self.adj_norm.to(device)
        else:
            A = torch.eye(self.num_nodes, device=device)

        if self.addaptadj:
            A_adapt = F.softmax(F.relu(self.nodevec1 @ self.nodevec2.T), dim=1)
            A = A + A_adapt

        return A

    def forward(self, x, adj=None):
        """
        Args:
            x: (batch, lookback, num_nodes) — speed values
            adj: (num_nodes, num_nodes) optional override
        Returns:
            (batch, horizon, num_nodes)
        """
        B, T, N = x.shape
        A = self._get_adj(adj, x.device)

        # Input projection: (B, T, N) -> (B, T, N, 1) -> (B, T, N, hidden)
        x = x.unsqueeze(-1)
        h = self.input_proj(x)
        h = F.relu(h)
        h = self.dropout(h)

        # Process through WaveNet layers
        skip_sum = 0

        for i in range(self.num_layers):
            dilation = 2 ** i
            pad_size = dilation

            # Reshape for temporal conv: (B, T, N, hidden) -> (B*N, hidden, T)
            h_temp = h.permute(0, 2, 3, 1).reshape(B * N, self.hidden_dim, T)

            # Causal padding (left only)
            h_temp = F.pad(h_temp, (pad_size, 0))

            # Gated temporal conv
            f = torch.tanh(self.filter_convs[i](h_temp))
            g = torch.sigmoid(self.gate_convs[i](h_temp))
            h_gated = f * g

            # Skip connection
            h_skip = h_gated[:, :, -1]
            skip_out = self.skip_convs[i](h_skip)
            skip_sum = skip_sum + skip_out

            # Spatial conv: (B*N, hidden, T) -> (B, T, N, hidden) -> matmul(A)
            h_spatial = h_gated.reshape(B, N, self.hidden_dim, T)
            h_spatial = h_spatial.permute(0, 3, 1, 2)
            h_spatial = torch.matmul(A, h_spatial)

            if i < self.num_layers - 1:
                h_spatial = F.relu(h_spatial)
                h = h + h_spatial
            else:
                h = h_spatial

        # Output from aggregated skip connections
        skip_agg = F.relu(skip_sum.reshape(B, N, -1))
        out = F.relu(self.end_conv_1(skip_agg))
        out = self.dropout(out)
        out = self.end_conv_2(out)

        return out.permute(0, 2, 1)

