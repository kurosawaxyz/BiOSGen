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

sys.path.insert(0, os.path.abspath('..'))

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
html_static_path = ['source/_static']
html_css_files = [
    'css/custom.css',
]


html_theme_options = {
    'analytics_id': 'G-XXXXXXXXXX',  #  Provided by Google in your dashboard
    'analytics_anonymize_ip': False,
    'logo_only': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#FFC2D1',
    'flyout_display': 'hidden',
    'version_selector': False,
    'language_selector': True,
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': False,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

html_logo = "assets/logo.png"
html_favicon = "assets/logo.png"