import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from domain_adaptation import GradientReversalLayer, GradientReversalFunction, DomainDiscriminator



class CrossAttentionFeatureFusion(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x1, x2):
        combined = torch.cat([x1, x2], dim=-1)   # [B, 2*hidden_dim]
        gate = self.gate(combined)               # [B, hidden_dim]
        fused = self.fusion(combined)            # [B, hidden_dim]
        return gate * x1 + (1 - gate) * fused


class CrossAttentionTransformer(nn.Module):
    def __init__(self, hidden_dim, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # x: [seq_len, batch_size, hidden_dim]
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x


class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, nhead, dropout=0.1):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value, key_padding_mask=None):
        """
        Cross-Attention
        Args:
            query: [seq_len_q, batch_size, hidden_dim]
            key_value: [seq_len_kv, batch_size, hidden_dim]
        """
        attn_output, _ = self.cross_attention(query, key_value, key_value, 
                                            key_padding_mask=key_padding_mask)
        return self.norm(query + self.dropout(attn_output))

class CrossAttentionPPIMI(nn.Module):
    def __init__(self, 
                 modulator_emb_dim,  
                 ppi_emb_dim,        
                 fingerprint_dim,    
                 nhead=4, 
                 num_cross_layers=2,
                 dim_feedforward=512, 
                 dropout=0.2,
                 grl_lambda=1.0
                 ):
        super().__init__()
        
        hidden_dim = 512

        self.modulator_proj = nn.Sequential(
            nn.Linear(modulator_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.fingerprint_proj = nn.Sequential(
            nn.Linear(fingerprint_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.ppi_proj = nn.Sequential(
            nn.Linear(ppi_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.mod_fp_cross_attention = nn.ModuleList([
            CrossAttentionLayer(hidden_dim, nhead, dropout)
            for _ in range(num_cross_layers)
        ])
        
        self.mod_ppi_cross_attention = nn.ModuleList([
            CrossAttentionLayer(hidden_dim, nhead, dropout)
            for _ in range(num_cross_layers)
        ])

        self.feature_fusion = CrossAttentionFeatureFusion(hidden_dim, dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )
        
        self.grl = GradientReversalLayer(lambda_=grl_lambda)
        self.domain_discriminator = DomainDiscriminator(hidden_dim)

    def forward(self, modulator, fingerprints, ppi_feats, domain_label=False):

        if modulator.dim() == 3:
            modulator = modulator.mean(dim=1)
        if fingerprints.dim() == 3:
            fingerprints = fingerprints.mean(dim=1)
        if ppi_feats.dim() == 3:
            ppi_feats = ppi_feats.mean(dim=1)


        mod_feats = self.modulator_proj(modulator)    # [B, hidden_dim]
        fp_feats = self.fingerprint_proj(fingerprints)
        ppi_feats = self.ppi_proj(ppi_feats)


        mod_feats = mod_feats.unsqueeze(0)  # [1, B, hidden_dim]
        fp_feats = fp_feats.unsqueeze(0)    # [1, B, hidden_dim]
        ppi_feats = ppi_feats.unsqueeze(0)  # [1, B, hidden_dim]


        for mod_fp_layer, mod_ppi_layer in zip(self.mod_fp_cross_attention, 
                                             self.mod_ppi_cross_attention):

            mod_feats = mod_fp_layer(mod_feats, fp_feats)
            mod_feats = mod_ppi_layer(mod_feats, ppi_feats)


        mod_feats = mod_feats.squeeze(0)  # [B, hidden_dim]
        fp_feats = fp_feats.squeeze(0)    # [B, hidden_dim]
        ppi_feats = ppi_feats.squeeze(0)  # [B, hidden_dim]


        fused_mod = self.feature_fusion(mod_feats, fp_feats)
        fused_feats = self.feature_fusion(fused_mod, ppi_feats)
        

        logits = self.classifier(fused_feats)


        if domain_label:
            reversed_feats = self.grl(fused_feats)
            domain_pred = self.domain_discriminator(reversed_feats)
            return logits, domain_pred
        else:
            l2_loss = 0.0
            for param in self.parameters():
                l2_loss += torch.norm(param)
            l2_loss = 0.0001 * l2_loss
            return logits, l2_loss 
        
        
class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.disc(x)