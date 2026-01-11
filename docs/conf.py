# Configuration file for the Sphinx documentation builder.
# FlowShield-UDRL Documentation

import os
import sys
from unittest.mock import MagicMock

# Mock heavy dependencies for ReadTheDocs
class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

MOCK_MODULES = ['torch', 'torch.nn', 'torch.optim', 'numpy', 'scipy', 
                'gymnasium', 'stable_baselines3', 'torchdyn', 'tensorboard',
                'tqdm', 'matplotlib', 'matplotlib.pyplot', 'PIL']
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'FlowShield-UDRL'
copyright = '2026, Serraji Wiam'
author = 'Serraji Wiam'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.githubpages',
    'sphinx_rtd_theme',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = '.rst'
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = None
html_favicon = None

html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'style_nav_header_background': '#2980B9',
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
    'vcs_pageview_mode': 'view',
}

# GitHub integration - "View on GitHub" points to repository
html_context = {
    'display_github': True,
    'github_user': 'MerlinMaven',
    'github_repo': 'FlowShield_UDRL',
    'github_version': 'main',
    'conf_py_path': '',
    'github_url': 'https://github.com/MerlinMaven/FlowShield_UDRL',
}

# -- Extension configuration -------------------------------------------------
# Napoleon settings (for Google/NumPy docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

# MathJax configuration
mathjax3_config = {
    'tex': {
        'macros': {
            'RR': r'\mathbb{R}',
            'EE': r'\mathbb{E}',
            'PP': r'\mathbb{P}',
            'LL': r'\mathcal{L}',
            'DD': r'\mathcal{D}',
            'NN': r'\mathcal{N}',
        }
    }
}

# Todo extension
todo_include_todos = True

# -- Custom CSS --------------------------------------------------------------
def setup(app):
    app.add_css_file('custom.css')
