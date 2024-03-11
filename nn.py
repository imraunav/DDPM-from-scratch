import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import math
from abc import abstractmethod


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class TimeEmbeddedBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        pass


class TimeEmbeddedSequential(nn.Sequential, TimeEmbeddedBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimeEmbeddedBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class ResBlock(TimeEmbeddedBlock):
    def __init__(
        self,
        channels,
        out_channels,
        time_emb_dim,
        groups=8,
        dropout=0,
        up=False,
        down=False,
    ):
        super().__init__()

        self.updown = up or down

        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        self.block1 = Block(channels, out_channels, groups)
        self.block2 = Block(out_channels, out_channels, groups, dropout)
        if channels != out_channels:
            self.skip_connection = nn.Conv2d(
                channels, out_channels, kernel_size=1, padding=0
            )
        else:
            self.skip_connection = nn.Identity()

        if up:
            scale = 2
        elif down:
            scale = 1 / 2
        else:
            scale = 1

        self.h_upd = nn.Upsample(scale_factor=scale)
        self.x_upd = nn.Upsample(scale_factor=scale)

    def forward(self, x, emb):
        time_emb = self.mlp(emb)
        h = self.block1(x)
        if self.updown:
            h = self.h_upd(h)
            x = self.x_upd(x)

        while len(h.shape) > len(time_emb.shape):
            time_emb = time_emb[..., None]

        h = h + time_emb

        return self.skip_connection(x) + self.block2(h)


class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        q, k, v = qkv.chunk(3, dim=1)
        ch = width // (3 * self.n_heads)
        scale = 1 / math.sqrt(math.sqrt(ch))

        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).contiguous().view(bs * self.n_heads, ch, length),
            (k * scale).contiguous().view(bs * self.n_heads, ch, length),
        )
        weight = torch.softmax(weight, dim=-1)
        a = torch.einsum(
            "bts,bcs->bct", weight, v.contiguous().view(bs * self.n_heads, ch, length)
        )
        return a.contiguous().view(bs, -1, length)


class AttentionBlock(nn.Module):
    def __init__(self, channels, n_heads, groups=8):
        super().__init__()
        self.n_heads = n_heads
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv1d(channels, 3 * channels, kernel_size=1)
        self.attention = QKVAttention(n_heads)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        bs, c, hight, width = x.shape
        x = x.reshape(bs, c, -1)
        h = self.norm(x)
        qkv = self.qkv(h)
        # print(qkv.shape)
        h = self.attention(qkv)
        # print(h.shape)
        h = self.proj(h)
        return (h + x).contiguous().view(bs, c, hight, width)


class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channel,
        out_channels,
        n_resblocks,
        n_heads,
        groups,
        dropout,
        n_classes=None,
        noise_steps=1000,
        attention_res=(16,),
        channel_mult=(1, 2, 4, 8),
    ):
        super().__init__()

        self.n_classes = n_classes
        # Add class embedding
        if n_classes is not None:
            self.class_emb = nn.Sequential(
                nn.Embedding(n_classes, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )

        time_emb_dim = model_channel
        self.time_emb = nn.Sequential(
            nn.Embedding(noise_steps, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        ch = model_channel * channel_mult[0]
        self.in_conv = nn.Conv2d(in_channels, ch, kernel_size=3, padding=1)
        skip_channels = [ch]

        # Init conv
        self.in_blocks = nn.ModuleList([])
        img_res = 256
        for level, mult in enumerate(channel_mult):
            for i in range(n_resblocks):
                layers = [
                    ResBlock(
                        ch, int(model_channel * mult), time_emb_dim, groups, dropout
                    )
                ]
                ch = int(model_channel * mult)
                if img_res in attention_res:
                    layers.append(AttentionBlock(ch, n_heads, groups))
                self.in_blocks.append(TimeEmbeddedSequential(*layers))
                skip_channels.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                layers = ResBlock(
                    ch,
                    int(model_channel * mult),
                    time_emb_dim,
                    groups,
                    dropout,
                    down=True,
                )
                img_res //= 2
                ch = int(model_channel * mult)
                self.in_blocks.append(layers)
                skip_channels.append(ch)

                ch = out_ch

        self.mid_blocks = TimeEmbeddedSequential(
            ResBlock(ch, ch, time_emb_dim, groups, dropout),
            AttentionBlock(ch, n_heads, groups),
            ResBlock(ch, ch, time_emb_dim, groups, dropout),
        )

        self.out_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(n_resblocks + 1):
                ich = skip_channels.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        int(model_channel * mult),
                        time_emb_dim,
                        groups,
                        dropout,
                    )
                ]
                ch = int(model_channel * mult)
                if img_res in attention_res:
                    layers.append(AttentionBlock(ch, n_heads, groups))

                if level and i == n_resblocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(ch, out_ch, time_emb_dim, groups, dropout, up=True)
                    )
                    img_res = int(img_res * 2)
                    ch = int(model_channel * mult)
                self.out_blocks.append(TimeEmbeddedSequential(*layers))

        self.out_conv = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x, timesteps, y=None):
        assert (y is not None) == (
            self.n_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        emb = self.time_emb(timesteps)

        if self.n_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = self.in_conv(x)
        hs = [h]
        for block in self.in_blocks:
            h = block(h, emb)
            hs.append(h)

        h = self.mid_blocks(h, emb)

        for block in self.out_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = block(h, emb)

        return self.out_conv(h)
