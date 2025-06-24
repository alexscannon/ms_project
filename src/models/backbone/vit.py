from typing import Dict
import torch
import torch.nn as nn
from omegaconf import DictConfig
from timm import create_model
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .SPT import ShiftedPatchTokenization

# Helper Functions
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

# Classes
class DropPath(nn.Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PreNorm(nn.Module):
    def __init__(self, num_tokens, dim, fn):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), ** kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, num_patches, hidden_dim, dropout = 0.):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, num_patches, heads = 8, dim_head = 64, dropout = 0., is_LSA=False):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.num_patches = num_patches
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim = dim
        self.inner_dim = inner_dim
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(self.dim, self.inner_dim * 3, bias = False)
        init_weights(self.to_qkv)
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, self.dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        if is_LSA:
            self.scale = nn.Parameter(self.scale*torch.ones(heads))
            self.mask = torch.eye(self.num_patches+1, self.num_patches+1)
            self.mask = torch.nonzero((self.mask == 1), as_tuple=False)
        else:
            self.mask = None

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        if self.mask is None:
            dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        else:
            scale = self.scale
            dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k), scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((b, h, 1, 1)))
            dots[:, :, self.mask[:, 0], self.mask[:, 1]] = -987654321

        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def flops(self):
        flops = 0
        if not self.is_coord:
            flops += self.dim * self.inner_dim * 3 * (self.num_patches+1)
        else:
            flops += (self.dim+2) * self.inner_dim * 3 * self.num_patches
            flops += self.dim * self.inner_dim * 3


class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim_ratio, dropout = 0., stochastic_depth=0., is_LSA=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.scale = {}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(num_patches, dim, Attention(dim, num_patches, heads = heads, dim_head = dim_head, dropout = dropout, is_LSA=is_LSA)),
                PreNorm(num_patches, dim, FeedForward(dim, num_patches, dim * mlp_dim_ratio, dropout = dropout))
            ]))
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()

    def forward(self, x):
        for i, (attn, ff) in enumerate(self.layers):
            x = self.drop_path(attn(x)) + x
            x = self.drop_path(ff(x)) + x
            self.scale[str(i)] = attn.fn.scale
        return x

class ViT(nn.Module):
    def __init__(self, *, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim_ratio, channels = 3,
                 dim_head = 16, dropout = 0.0, emb_dropout = 0.0, stochastic_depth=0.0, is_LSA=False,
                 is_SPT=False):
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = channels * patch_height * patch_width
        self.dim = dim
        self.num_classes = num_classes

        if not is_SPT:
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
                nn.Linear(self.patch_dim, self.dim)
            )

        else:
            self.to_patch_embedding = ShiftedPatchTokenization(3, self.dim, patch_size, is_pe=True)

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(self.dim, self.num_patches, depth, heads, dim_head, mlp_dim_ratio, dropout,
                                       stochastic_depth, is_LSA=is_LSA)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_classes)
        )

        self.apply(init_weights)

    def forward(self, img):
        # patch embedding

        x = self.to_patch_embedding(img)

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)

        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        return self.mlp_head(x[:, 0])






#  ========= OLD ViT Class
class VisionTransformer(nn.Module):
    def __init__(self, config: DictConfig, checkpoint_data: Dict):
        super().__init__()

        self.config = config
        self.dataset_name = config.data.name
        self.device = config.device
        self.model_name = self.config.model.backbone.get('name', 'vit_small_patch16_224')
        self.checkpoint_data = checkpoint_data

        self.model = self.load_pretrained_model() # Renamed 'location' to 'checkpoint_type' for clarity if it means 'best', 'last' etc.

    def load_pretrained_model(self) -> nn.Module:
        """
        Load a pretrained ViT model from the location specified in the config.
        """
        # Load the checkpoint data
        model_state_dict_key = self.config.model.backbone.checkpoint_keys.get('model_state_dict', 'model_state_dict')

        # Load the model state dictionary
        state_dict = self.checkpoint_data[model_state_dict_key]

        # --- Model Creation & Loading ---
        try:
            model = self.create_raw_vit(num_classes=self.config.model.backbone.num_classes)
            if not self.config.model.pretrained:
                return model # Do not load pretrained weights if user wants raw ViT

            model.load_state_dict(state_dict) # load the pre-trained ViT parameters
            model.to(self.device) # Ensure model is on the correct device

            logging.info(f"Loaded {self.dataset_name} ViT model with {self.config.model.backbone.num_classes} number of classes ...")
            return model # Return the loaded model

        except RuntimeError as e:
            logging.error(f"Failed to load state dict into model: {e}")
            raise RuntimeError(f"Failed to load state dict into model: {e}")


    def create_raw_vit(self, num_classes: int) -> nn.Module:
        """
        Create a raw ViT model.
        Args:
            num_classes (int): Number of output classes
        Returns:
            model (nn.Module): ViT model
        """
        # If num_classes=0, this removes classification head, allowing model to be used as a feature extractor
        try:
            model = create_model(
                model_name=self.model_name,
                pretrained=False,
                num_classes=num_classes,
                patch_size=self.config.model.backbone.patch_size,
                drop_path_rate=self.config.model.backbone.dropout_rate,
            )
            logging.info(f"Created raw ViT model '{self.model_name}' with {num_classes} output classes.")
            return model
        except Exception as e:
            logging.error(f"Failed to create raw ViT model: {e}")
            raise RuntimeError(f"Failed to create raw ViT model: {e}")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ViT model.
        Args:
            x (torch.Tensor): Input tensor
        Returns:
            logits (torch.Tensor): Logits from the model
        """
        logits = self.model(x)
        return logits

