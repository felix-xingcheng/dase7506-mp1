"""Working classroom baseline: a small GPT trained from random initialization."""
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, width=128, heads=4):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)  #归一化
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width) #qkv输出3*width是为了一次性表示qkv，然后拆分为q,k,v
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))#先升维增强表示性，做特征层后再降维恢复原始维度

    def forward(self, x):#算qkv矩阵然后计算注意力分数
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)#view将width按qkv以及头的长度拆分，permute调换顺序，方便分离qkv
        # Each position attends only to itself and earlier input tokens.
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)#集成的attention函数，在内部计算att分数并与V相乘出结果
        x = x + self.proj(attended.transpose(1, 2).reshape(batch, length, width))#将注意力分数与原始输入相加，再通过投影层进一步处理
        return x + self.mlp(self.norm2(x))#先归一化再做第二次残差求和


class GPT(nn.Module):
    def __init__(self, config):#config："vocab": 2048,"width": 128,"heads": 4,"depth": 4,"context": 256
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        self.token = nn.Embedding(config['vocab'], width)
        self.pos = nn.Embedding(self.context, width)#为什么这样写可以起到位置编码的效果
        self.blocks = nn.ModuleList([Block(width, config['heads']) for _ in range(config['depth'])])#重复四遍
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self.initialize)
        self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):#初始化权重
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):#编码+四遍decoding(block)
        x = self.token(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))#pos这里提取ids的token数量，然后加上位置编码
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
