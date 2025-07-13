import os
import sys
from unittest.mock import MagicMock

# Enhanced Mock class
class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

MOCK_MODULES = [
    "torch", "torchvision", "torchaudio", "torch.nn", "torch.utils",
    "torch.utils.data", "torch.optim", "torch.nn.functional",
    "flash_attn", "clip", "transformers", "diffusers",
    "loralib", "peft", "pytorch_fid", "cleanfid",
    "thop", "torchviz", "torchinfo",
    "PIL", "cv2", "accelerate"
]

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = Mock()

sys.path.insert(0, os.path.abspath('../..'))

# Project information
project = 'BiOSGen'
author = 'H. T. Duong Vu'
release = '0.1.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

# Autodoc configuration
autodoc_mock_imports = [
    "torch", "torchvision", "torchaudio",
    "flash_attn", "clip", "transformers", "diffusers",
    "loralib", "peft", "pytorch_fid", "cleanfid",
    "thop", "torchviz", "torchinfo",
    "PIL", "cv2", "accelerate"
]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# Template and static paths
templates_path = ['_templates']
exclude_patterns = []

# HTML theme
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']