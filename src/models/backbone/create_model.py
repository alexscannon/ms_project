from omegaconf import DictConfig
from .vit import ViT


def create_model(img_size: int, n_classes: int, config: DictConfig):
    if config.model.backbone.name == 'vit':
        patch_size = 4 if img_size == 32 else 8
        dim_head = config.model.backbone.dim // config.model.backbone.heads

        model = ViT(
            img_size=img_size,
            patch_size=patch_size,
            num_classes=n_classes,
            mlp_dim_ratio=config.model.backbone.mlp_dim_ratio,
            depth=config.model.backbone.depth,
            dim=config.model.backbone.dim,
            heads=config.model.backbone.heads,
            dim_head=dim_head,
            stochastic_depth=config.model.backbone.stochastic_depth,
            is_SPT=config.model.backbone.is_SPT,
            is_LSA=config.model.backbone.is_LSA
        )

    return model