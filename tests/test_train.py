import pytest
import json
import re
import hydra
from pathlib import Path
import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Import train function
from src.train import train
import logging
from datetime import datetime
import time


@pytest.fixture
def config():
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(
            config_name="train",
            overrides=["experiment=dogbreed_ex_train","trainer.max_epochs=3"],
        )
        return cfg

def parse_metrics_from_console_output(caplog):
    """Parse metrics from the captured console output."""
    for record in caplog.records:
        if "'val_acc':" in record.getMessage():
            # Extract the dictionary-like string
            metrics_str = re.search(r'{.*}', record.getMessage()).group(0)
            
            # Parse individual values using regex
            metrics = {}
            pattern = r"'(\w+)': tensor\(([\d.]+)\)"
            matches = re.finditer(pattern, metrics_str)
            
            for match in matches:
                key = match.group(1)
                value = float(match.group(2))
                metrics[key] = value
                
            return metrics
    return None

@pytest.fixture
def caplog(caplog):
    """Fixture to ensure caplog captures the right log level"""
    caplog.set_level(logging.INFO)
    return caplog

@pytest.mark.order(1)
def test_dogbreed_ex_training(config, tmp_path, caplog):
    # Update output and log directories to use temporary path
    config.paths.output_dir = str(tmp_path)
    config.paths.log_dir = str(tmp_path / "logs")
    
    # Run training
    train(config)
    
    # Parse metrics from console output
    metrics = parse_metrics_from_console_output(caplog)
    
    # Debug output
    print("All captured logs:")
    for record in caplog.records:
        print(record.getMessage())
    print(f"Parsed metrics: {metrics}")
    
    # Assert metrics were found and validation accuracy meets threshold
    assert metrics is not None, "Could not find metrics in console output"
    val_acc = metrics['val_acc']
    assert val_acc > 0.30, f"Validation accuracy {val_acc} is not greater than 0.30"
