"""
Transformer-based neural surrogate for Rydberg facilitation dynamics.
Predicts the full trajectory in a single forward pass via direct regression.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Multi-head self-attention without causal masking (full bidirectional)."""
    
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))
        return y


class TransformerBlock(nn.Module):
    """Standard transformer block: attention -> MLP with residuals."""
    
    def __init__(self, n_embd, n_head, dropout=0.1, mlp_ratio=2):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_ratio * n_embd),
            nn.GELU(),
            nn.Linear(mlp_ratio * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class RydbergSurrogate(nn.Module):
    """Transformer-based surrogate predicting sz_mean(t) in one forward pass."""
    
    def __init__(self, n_layer=4, n_head=4, n_embd=96, n_time=400, dropout=0.2, mlp_ratio=2):
        super().__init__()
        self.n_embd = n_embd
        self.n_time = n_time
        
        # Parameter embedding: [Omega, N, 1/sqrt(N), log10(gamma), dimension] -> n_embd
        # Input normalization constants (approx dataset stats for zero-mean unit-variance)
        self.register_buffer('param_mean', torch.tensor([15.0, 2000.0, 0.04, 0.0, 2.0]))
        self.register_buffer('param_std', torch.tensor([10.0, 2000.0, 0.02, 1.0, 1.0]))
        
        self.param_embed = nn.Sequential(
            nn.Linear(5, n_embd),
            nn.LayerNorm(n_embd),
            nn.GELU(),
            nn.Linear(n_embd, n_embd),
        )
        
        # Time coordinate embedding
        self.time_embed = nn.Linear(1, n_embd)
        self.register_buffer('t_mean', torch.tensor(500.0))
        self.register_buffer('t_std', torch.tensor(300.0))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, dropout, mlp_ratio)
            for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, 1)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, omega, n_atoms, inv_sqrt_n, gamma, dimension, t):
        """
        Args:
            omega: (batch,) or (batch, 1)
            n_atoms: (batch,) or (batch, 1)
            inv_sqrt_n: (batch,) or (batch, 1)
            gamma: (batch,) or (batch, 1) — log10(γ)
            dimension: (batch,) or (batch, 1) — 1.0, 2.0, or 3.0
            t: (batch, n_time)
            
        Returns:
            sz_pred: (batch, n_time) predicted sz_mean values
        """
        batch_size = t.size(0)
        
        if omega.dim() == 1:
            omega = omega.unsqueeze(1)
        if n_atoms.dim() == 1:
            n_atoms = n_atoms.unsqueeze(1)
        if inv_sqrt_n.dim() == 1:
            inv_sqrt_n = inv_sqrt_n.unsqueeze(1)
        if gamma.dim() == 1:
            gamma = gamma.unsqueeze(1)
        if dimension.dim() == 1:
            dimension = dimension.unsqueeze(1)
        
        params = torch.cat([omega, n_atoms, inv_sqrt_n, gamma, dimension], dim=-1)
        params_norm = (params - self.param_mean) / self.param_std
        param_emb = self.param_embed(params_norm)  # (batch, n_embd)
        
        t_reshaped = t.unsqueeze(-1)  # (batch, n_time, 1)
        t_norm = (t_reshaped - self.t_mean) / self.t_std
        time_emb = self.time_embed(t_norm)  # (batch, n_time, n_embd)
        
        x = param_emb.unsqueeze(1) + time_emb  # (batch, n_time, n_embd)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        sz_pred = self.head(x).squeeze(-1)  # (batch, n_time)
        
        return sz_pred
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = RydbergSurrogate(n_layer=4, n_head=4, n_embd=96, mlp_ratio=2)
    print(f"Model parameters: {model.count_parameters():,}")
    
    batch_size = 4
    omega = torch.randn(batch_size)
    n_atoms = torch.tensor([225.0, 400.0, 900.0, 1225.0])
    inv_sqrt_n = 1.0 / torch.sqrt(n_atoms)
    gamma = torch.tensor([-1.0, -1.0, -1.0, -1.0])  # log10(0.1)
    dimension = torch.tensor([2.0, 2.0, 2.0, 2.0])
    t = torch.linspace(0, 1000, 400).unsqueeze(0).expand(batch_size, -1)
    
    sz_pred = model(omega, n_atoms, inv_sqrt_n, gamma, dimension, t)
    print(f"Input shapes: omega={omega.shape}, t={t.shape}")
    print(f"Output shape: sz_pred={sz_pred.shape}")
