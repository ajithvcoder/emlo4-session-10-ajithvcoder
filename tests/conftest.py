import pytest
from omegaconf import OmegaConf


@pytest.fixture
def config():
    return OmegaConf.load("configs/eval.yaml")


def test_dogbreed_eval(config):
    assert config.task_name == "eval"
    assert config.train == False


@pytest.fixture
def config_infer():
    return OmegaConf.load("configs/infer.yaml")

