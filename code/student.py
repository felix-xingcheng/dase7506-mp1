"""Your algorithm goes here. The default is a complete, runnable baseline.

Required work: diagnose a limitation and implement a structural/training/memory
change. Explain it, measure its cost and perform a mechanism ablation. Merely
renaming the baseline or reporting a lucky seed is not an algorithmic contribution.
You can replace this factory/model completely while keeping the two model interfaces.
"""
import torch
from torch import nn
from torch.nn import functional as F

class Block(nn.Module):
    def __init__(self, width=128, heads=4, dropout=0.0, hidden_multiplier=8 / 3):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)  #归一化
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width) #qkv输出3*width是为了一次性表示qkv，然后拆分为q,k,v
        self.mlp = GatedMLP(width, hidden_multiplier)
        self.residual_dropout = nn.Dropout(dropout)

    def apply_rope(self, x):
        # x: [batch, heads, length, head_dim]
        d = x.shape[-1]
        x_float = x.float()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
        pos = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos(), emb.sin()
        x1, x2 = x_float[..., : d // 2], x_float[..., d // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x_float * cos + rotated * sin).to(dtype=x.dtype)

    def forward(self, x):#算qkv矩阵然后计算注意力分数
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)#view将width按qkv以及头的长度拆分，permute调换顺序，方便分离qkv
        # Each position attends only to itself and earlier input tokens.
        q, k = self.apply_rope(q), self.apply_rope(k)  
        attended = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )#集成的attention函数，在内部计算att分数并与V相乘出结果
        x = x + self.residual_dropout(self.proj(attended.transpose(1, 2).reshape(batch, length, width)))#将注意力结果与原始输入相加，再通过投影层进一步处理
        return x + self.residual_dropout(self.mlp(self.norm2(x)))#先归一化再做第二次残差求和

class GatedMLP(nn.Module):
    def __init__(self, width, hidden_multiplier=8 / 3):
        super().__init__()
        hidden = int(hidden_multiplier * width)
        self.up = nn.Linear(width, hidden)
        self.gate = nn.Linear(width, hidden)
        self.down = nn.Linear(hidden, width)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class GPT(nn.Module):
    def __init__(self, config):#config："vocab": 2048,"width": 128,"heads": 4,"depth": 4,"context": 256
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        dropout = config.get('dropout', 0.0)
        hidden_multiplier = config.get('hidden_multiplier', 8 / 3)
        self.token = nn.Embedding(config['vocab'], width)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(width, config['heads'], dropout, hidden_multiplier)
            for _ in range(config['depth'])
        ])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=True)
        self.apply(self.initialize)

    @staticmethod
    def initialize(module):#初始化权重
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)


    def features(self, ids):#编码+四遍decoding(block)
        x = self.embedding_dropout(self.token(ids))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids):
        """Training interface: unnormalized next-token logits [batch, time, vocab]."""
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        """Evaluation interface: normalized log probabilities, with no access to targets.

        Override this for a strictly causal, within-window memory mechanism.
        A prediction at position t can use ids[:, :t+1] and nothing later.
        Reset all temporary state on every call; each evaluation window starts fresh.
        """
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    """Keep the classroom baseline runnable with --implementation model."""
    #这样可以使用GPT的所有函数，但无法调用class block的函数，block的函数只能在GPT内部执行
    return GPT(config)


