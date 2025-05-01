import torch
import torch.nn as nn

class AttentionFusion(nn.Module):
    """
    Attention-based fusion module.
    Uses cross-attention: one stream attends to the other.
    """
    def __init__(self, feature_dim=128, fusion_dim=128, num_heads=4):
        super(AttentionFusion, self).__init__()
        # MultiheadAttention layers (feature_dim is the query/key dimension)
        self.attn1 = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        # Linear layer to combine both attended outputs
        self.fc = nn.Linear(feature_dim * 2, fusion_dim)
        
    def forward(self, feat_w, feat_t):
        """
        feat_w: (batch, feature_dim) from Whisper branch
        feat_t: (batch, feature_dim) from Traditional branch
        """
        # Add a sequence dimension for attention: (batch, 1, feature_dim)
        w = feat_w.unsqueeze(1)
        t = feat_t.unsqueeze(1)
        # Whisper as query, traditional as key+value
        attn_w, _ = self.attn1(w, t, t)  # -> (batch, 1, feature_dim)
        # Traditional as query, whisper as key+value
        attn_t, _ = self.attn2(t, w, w)  # -> (batch, 1, feature_dim)
        # Remove sequence dimension
        attn_w = attn_w.squeeze(1)       # -> (batch, feature_dim)
        attn_t = attn_t.squeeze(1)       # -> (batch, feature_dim)
        # Concatenate and project
        fused = torch.cat([attn_w, attn_t], dim=1)  # (batch, 2*feature_dim)
        fused = self.fc(fused)                       # (batch, fusion_dim)
        return fused
