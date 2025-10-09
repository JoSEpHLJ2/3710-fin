# modules.py
# VQ-VAE2 简化实现（2-level quantization）
# 说明：此实现为教学与实验目的，非生产级高优化实现，但足够用于课程作业与研究原型。

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# 基础卷积块
# -----------------------
class ConvRelu(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))

# -----------------------
# 下采样块（Conv stride=2）
# -----------------------
class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            ConvRelu(in_ch, out_ch, k=4, stride=2, padding=1),
            ConvRelu(out_ch, out_ch)
        )
    def forward(self, x): return self.net(x)

# -----------------------
# 上采样块（ConvTranspose）
# -----------------------
class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ConvRelu(out_ch, out_ch)
        )
    def forward(self, x): return self.net(x)

# -----------------------
# 向量量化模块（简单非EMA版本）
# -----------------------
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        # embeddings: [K, D]
        self.embeddings = nn.Parameter(torch.randn(num_embeddings, embedding_dim) * 0.01)

    def forward(self, z):
        """
        z: [B, C, H, W]  (C == embedding_dim)
        returns:
          quantized: [B, C, H, W]
          vq_loss: scalar
          encodings: [B*H*W, K] one-hot
        """
        B, C, H, W = z.shape
        assert C == self.embedding_dim, "embedding_dim mismatch"
        flat_z = z.permute(0,2,3,1).contiguous().view(-1, C)  # [N, C]
        # distances: [N, K] = ||z||^2 - 2 z e^T + ||e||^2
        dists = (flat_z.pow(2).sum(dim=1, keepdim=True)
                 - 2 * torch.matmul(flat_z, self.embeddings.t())
                 + self.embeddings.pow(2).sum(dim=1).unsqueeze(0))
        encoding_indices = torch.argmin(dists, dim=1)  # [N]
        encodings = F.one_hot(encoding_indices, num_classes=self.num_embeddings).type(flat_z.dtype)  # [N, K]
        quantized_flat = torch.matmul(encodings, self.embeddings)  # [N, C]
        quantized = quantized_flat.view(B, H, W, C).permute(0,3,1,2).contiguous()

        # losses
        q_loss = F.mse_loss(quantized.detach(), z)
        e_loss = F.mse_loss(quantized, z.detach())
        loss = q_loss + self.commitment_cost * e_loss

        # straight-through estimator
        quantized = z + (quantized - z).detach()

        # perplexity
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(- (avg_probs * torch.log(avg_probs + 1e-10)).sum())

        return quantized, loss, perplexity

# -----------------------
# Encoder 和 Decoder （为 VQ-VAE2 设计）
# -----------------------
class EncoderBottom(nn.Module):
    """
    bottom encoder：从输入逐级下采样到较低尺度（用于 bottom latent）
    """
    def __init__(self, in_ch=1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            ConvRelu(in_ch, hidden//2),
            Down(hidden//2, hidden//2),   # /2
            Down(hidden//2, hidden),      # /4
            ConvRelu(hidden, hidden)
        )
    def forward(self, x):
        return self.net(x)  # [B, hidden, H/4, W/4]

class EncoderTop(nn.Module):
    """
    top encoder：接 bottom 编码进一步下采样，得到更低分辨率的 top latent
    """
    def __init__(self, in_ch, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            Down(in_ch, in_ch),           # /8 relative to input
            ConvRelu(in_ch, hidden)
        )
    def forward(self, x):
        return self.net(x)  # [B, hidden, H/8, W/8]

class DecoderTop(nn.Module):
    """
    从 top quantized 恢复到 bottom 分辨率并与 bottom quantized 融合
    """
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            ConvRelu(hidden, hidden),
            Up(hidden, hidden),  # up from /8 -> /4
        )
    def forward(self, z_top):
        return self.net(z_top)  # [B, hidden, H/4, W/4]

class DecoderFinal(nn.Module):
    """
    最终 decoder：结合 bottom 特征（quantized bottom + upsampled top）逐步上采样到原始分辨率
    """
    def __init__(self, out_ch=1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            ConvRelu(hidden*2, hidden),
            Up(hidden, hidden//2),  # /2
            Up(hidden//2, hidden//4),
            ConvRelu(hidden//4, hidden//4),
            nn.Conv2d(hidden//4, out_ch, kernel_size=3, padding=1)
        )
    def forward(self, x):
        # x expected [B, hidden*2, H/4, W/4]
        return torch.sigmoid(self.net(x))

# -----------------------
# VQ-VAE2 wrapper
# -----------------------
class VQVAE2(nn.Module):
    def __init__(self, in_ch=1, hidden=128, num_embed_top=512, num_embed_bottom=512, commitment_cost=0.25):
        super().__init__()
        self.enc_bottom = EncoderBottom(in_ch=in_ch, hidden=hidden)
        self.enc_top = EncoderTop(in_ch=hidden, hidden=hidden)
        # convs to map to embedding_dim
        self.top_pre_vq = nn.Conv2d(hidden, hidden, kernel_size=1)
        self.bottom_pre_vq = nn.Conv2d(hidden, hidden, kernel_size=1)

        self.vq_top = VectorQuantizer(num_embed_top, hidden, commitment_cost)
        self.vq_bottom = VectorQuantizer(num_embed_bottom, hidden, commitment_cost)

        self.dec_top = DecoderTop(hidden=hidden)
        self.dec_final = DecoderFinal(out_ch=in_ch, hidden=hidden)

    def forward(self, x):
        # x: [B, C, H, W]
        z_b = self.enc_bottom(x)         # [B, hidden, H/4, W/4]
        z_t = self.enc_top(z_b)         # [B, hidden, H/8, W/8]
        z_t = self.top_pre_vq(z_t)
        # quantize top
        z_t_q, loss_t, perp_t = self.vq_top(z_t)
        # decode top to bottom resolution
        z_t_up = self.dec_top(z_t_q)    # [B, hidden, H/4, W/4]

        # prepare bottom codes: concatenate bottom encoder output with top decoded features
        z_b_pre = self.bottom_pre_vq(z_b)
        # combine
        z_b_comb = z_b_pre + z_t_up     # residual-like fusion
        z_b_q, loss_b, perp_b = self.vq_bottom(z_b_comb)
        # prepare final decoder input: concat quantized bottom and upsampled top (or use z_b_q with z_t_up)
        dec_in = torch.cat([z_b_q, z_t_up], dim=1)  # [B, hidden*2, H/4, W/4]
        x_recon = self.dec_final(dec_in)

        vq_loss = loss_t + loss_b
        perplexity = (perp_t + perp_b) / 2.0
        return x_recon, vq_loss, perplexity
