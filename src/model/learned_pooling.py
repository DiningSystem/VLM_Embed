import torch
import torch.nn as nn

class LearnedPooler(nn.Module):
    def __init__(self, hidden_dim, proj_dim=1024, num_latents=1, num_heads=8):
        super().__init__()
        self.num_latents = num_latents
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(hidden_dim, proj_dim)
        self.norm = nn.LayerNorm(proj_dim)
        # The Learnable Latent Queries
        
        self.latent_queries = nn.Parameter(torch.randn(1, num_latents, proj_dim))
        
        # Cross-Attention Layer
        # batch_first=True ensures inputs are (Batch, Seq, Dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=proj_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        
        self.mlp = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.Tanh(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, vlm_hidden_states, attention_mask=None):
        """
        vlm_hidden_states: (Batch, Seq_Len, Dim) - Output from the VLM
        attention_mask: (Batch, Seq_Len) - Standard padding mask (0 for padding)
        """
        hidden_states = self.proj(vlm_hidden_states)
        hidden_states = self.norm(hidden_states)
        batch_size = vlm_hidden_states.size(0)
        
        # Expand latents to match batch size: (Batch, Num_Latents, Dim)
        latents = self.latent_queries.expand(batch_size, -1, -1)
        
        # Invert attention mask for MultiheadAttention (True = Ignore) if provided
        # HuggingFace masks are usually 1=Keep, 0=Pad. PyTorch MHA wants True=Ignore.
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        
        pooled_output, _ = self.cross_attention(
            query=latents,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask
        )
        
        # If we only want one vector per sample, squeeze it
        if self.num_latents == 1:
            pooled_output = pooled_output.squeeze(1)
            
        # Optional projection
        return self.mlp(pooled_output)