import hydra
from omegaconf import DictConfig, OmegaConf

from src.models.backbone.vit import VisionTransformer

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    model = VisionTransformer(config=config)


if __name__ == "__main__":
    main()