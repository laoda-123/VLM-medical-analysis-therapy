import torch.nn as nn
import torch
import torch.nn.functional as F
import math
# use conservative  encoding the words 
class SineCosinePositionalEncoding(nn.Module):
    """正余弦位置编码层，支持分类token（+1维度）"""
    def __init__(self, embed_dim, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len=max_len
        self.embed_dim=embed_dim
        # 生成频率因子
        
        
        # 添加分类token的位置编码（固定为全零或其他值, 或者直接不使用)
        # 注意：这里的+1是为了支持(num_patches + 1)的输入形状
        # self.register_buffer('pe', pe.unsqueeze(0))  # 形状为 (1, max_len, embed_dim)

    def forward(self, x):
        """
        Args:
            x: 输入张量，形状为 (batch_size, num_patches + 1, embed_dim)
            
        Returns:
            torch.Tensor: 添加位置编码后的张量，形状与输入相同
        """
        batch_size,seq_len,embed_dim=x.shape[0],x.shape[1],x.shape[2]
       
        # 获取有效序列长度（包括分类token）
        position = torch.arange(self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2) * (-math.log(10000.0) / self.embed_dim))
        
        pe = torch.zeros(self.max_len, self.embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度用sin
        pe[:, 1::2] = torch.cos(position * div_term) 
        # 形状为 (1, seq_len, embed_dim)
        pe=pe.unsqueeze(0).repeat(batch_size,1,1)

        x=x+pe[:,:seq_len,:].to(x.device)
        # 添加位置编码并应用Dropout
        
        return self.dropout(x)
# Salesforce/blip2-flan-t5-xxl use the same positional encoding as the original transformer
class PatchEmbedding(nn.Module):
    """Patch嵌入层"""
    def __init__(self,embed_dim, stride,in_channels=3, patch_size=14):
        super().__init__()
        self.embed_dim=embed_dim
        self.stride=stride
        self.in_channels=in_channels
        self.patch_size= patch_size
        self.projection= nn.Conv2d(self.in_channels, self.embed_dim, self.patch_size, self.patch_size,0)
        self.cls_token=nn.Parameter(torch.zeros(1,1,embed_dim))
    def forward(self, x):
        """
        Args:
            x: 输入图像张量，形状为 (batch_size, in_channels, height, width)
            
        Returns:
            out: 输出patch序列,形状为 (batch_size, num_patches, embed_dim)
        """
        
        x= self.projection(x)  # (batch_size, embed_dim, num_patches_h, num_patches_w)
        x= x.flatten(2)  # (batch_size, embed_dim, num_patches)
        result= x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        
        result=torch.cat([result,self.cls_token.expand(result.shape[0],-1,-1)],dim=1) # if users want to annotate the image
        return result


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1, norm_first=False):
        super().__init__()
        
        self.norm_first= norm_first
        
        # 自注意力子层
        self.norm1= nn.LayerNorm(embed_dim)
        self.attn=nn.MultiheadAttention(embed_dim=embed_dim,num_heads=num_heads,bias=True)
        self.dropout1= nn.Dropout(p=dropout)
        
        # 前馈子层
        self.norm2= nn.LayerNorm(embed_dim)
        self.ffn= nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(p=dropout)
        )
        
    def forward(self, x, mask=None):
        """
        Args:
            x: 输入序列，形状为 (bartch_size, num_patches, embed_dim)
            mask: 注意力掩码
            
        Returns:
            out: 输出序列，形状为 (batch, num_patches, embed_dim)
        """
        if self.norm_first:
            # Pre-norm架构
            x= x + self._sa_block(self.norm1(x))
            x= x + self._ffn_block(self.norm2(x))
        else:
            # Post-norm架构
            x= self.norm1(x + self._sa_block(x))
            x= self.norm2(x + self._ffn_block(x))
            
        return x

    def _sa_block(self, x):
        """自注意力块"""
        x=self.attn(x,x,x)
        return self.dropout1(x)
    
    def _ffn_block(self,x):
        """前馈块"""
        x= self.ffn(x)
        return x






        
