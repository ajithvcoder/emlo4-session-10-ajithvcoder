import pytest
import hydra
from pathlib import Path

import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Import train function
from src.infer import infer


@pytest.fixture
def config():
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(
            config_name="infer",
            overrides=["callbacks.model_checkpoint.filename=/workspace/model_storage/epoch-checkpoint.ckpt.ckpt"],
        )
        return cfg

@pytest.mark.dependency(on=['tests/test_eval.py'])
@pytest.mark.order(3)
def test_dogbreed_ex_infer(config, tmp_path):
    # Set the path for infer_images
    infer_images_dir = Path("infer_images")
    
    # Delete all files in infer_images/* if any exist
    if infer_images_dir.exists() and infer_images_dir.is_dir():
        for file in infer_images_dir.glob("*"):
            file.unlink()  # Delete the file
    
    # Assert that infer_images directory is now empty
    assert len(list(infer_images_dir.glob("*"))) == 0, "infer_images directory is not empty before infer()"
    
    # Update output and log directories to use temporary path
    config.paths.output_dir = str(tmp_path)
    config.paths.log_dir = str(tmp_path / "logs")
    
    # Run inference
    infer(config)
    
    # Ensure there are 5 files in infer_images after infer()
    assert len(list(infer_images_dir.glob("*"))) == 5, "There are not exactly 5 files in infer_images after infer()"