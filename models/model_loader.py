import importlib.util
from pathlib import Path


class ModelLoader:
    """
    A class for loading models from a specified directory.

    Attributes:
        model_path (Path): pathlib.Path to the directory where models are stored. Defaults to the 'models' directory in the current working directory.

    Methods:
        get_model(cfg, embeddings): Loads and returns a model based on the provided configuration and embeddings.
    """

    model_path = Path.cwd() / 'models'

    @classmethod
    def get_model(cls, cfg, embeddings):
        model_name = cfg['model_name']
        assert model_name in [path.stem for path in cls.model_path.iterdir()], f"Model {model_name} not found in {cls.model_path}"
        # Load the given model
        module_path = cls.model_path / (model_name + ".py")
        spec = importlib.util.spec_from_file_location(model_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.get_model(cfg, embeddings)