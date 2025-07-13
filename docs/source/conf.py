import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath('..'))  # so Sphinx can find your package

project = 'BiOSGen'
author = 'H. T. Duong Vu'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',        # For Google/NumPy-style docstrings
    'sphinx.ext.viewcode',        # Adds links to highlighted source code
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

MOCK_MODULES = [
    "torch", "torchvision", "torchaudio",
    "flash_attn", "clip", "transformers", "diffusers",
    "loralib", "peft", "pytorch_fid", "cleanfid",
    "thop", "torchviz", "torchinfo"
]

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = MagicMock()
