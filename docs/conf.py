# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))
#for x in os.walk('../peebee'):
#  sys.path.insert(0, x[0])

#this prevents imports from showing up in the docs
def patch_automodapi(app):
    """Monkey-patch the automodapi extension to exclude imported members"""
    from sphinx_automodapi import automodsumm
    from sphinx_automodapi.utils import find_mod_objs
    automodsumm.find_mod_objs = lambda *args: find_mod_objs(args[0], onlylocals=True)

def setup(app):
    app.connect("builder-inited", patch_automodapi)

#this fixes math dollar not working in autosum tables
from docutils.nodes import FixedTextElement, literal,math
from docutils.nodes import  comment, doctest_block, image, literal_block, math_block, paragraph, pending, raw, rubric, substitution_definition, target
math_dollar_node_blacklist = (literal,math,doctest_block, image, literal_block,  math_block,  pending,  raw,rubric, substitution_definition,target)

mathjax_config = {
    'tex2jax': {
        'inlineMath': [ ["\\(","\\)"] ],
        'displayMath': [["\\[","\\]"] ],
    },
}

mathjax3_config = {
  "tex": {
    "inlineMath": [['\\(', '\\)']],
    "displayMath": [["\\[", "\\]"]],
  }
}

#controls the sidebar
#html_sidebars = {
#   '**': ['text', 'globaltoc.html', 'searchbox.html'],
#}

#some options for the html theme
html_theme_options = {
    'fixed_sidebar': True,
    'description': "A python package merging Galactic dynamics with direct acceleration measurements",
    'page_width': '80%',
}

project = 'peebee'
copyright = '2024, Tom Donlon'
author = 'Tom Donlon'
release = '1.2.0' #TODO: read this from a file

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx_automodapi.automodapi',
              'sphinx_math_dollar',
              'sphinx.ext.imgmath',
              'sphinx_toolbox.sidebar_links']
numpydoc_show_class_members = False
              
#removed sphinx.ext.imgmath

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

