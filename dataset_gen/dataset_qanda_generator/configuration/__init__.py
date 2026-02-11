# dataset_gen/dataset_qanda_generator/configuration/__init__.py

from .env_adapter import load_global_env_for_wsl
load_global_env_for_wsl()

from .config import Config

cfg = Config()

__all__ = ["cfg", "Config"]
