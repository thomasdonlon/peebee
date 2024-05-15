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

project = 'peebee'
copyright = '2024, Tom Donlon'
author = 'Tom Donlon'
release = '0.0.8' #TODO: read this from a file

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.imgmath',
              'sphinx_automodapi.automodapi']
numpydoc_show_class_members = False
              
#removed sphinx_math_dollar
#removed sphinx.ext.imgmath

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

#this prevents imports from showing up in the docs
def patch_automodapi(app):
    """Monkey-patch the automodapi extension to exclude imported members"""
    from sphinx_automodapi import automodsumm
    from sphinx_automodapi.utils import find_mod_objs
    automodsumm.find_mod_objs = lambda *args: find_mod_objs(args[0], onlylocals=True)

def setup(app):
    app.connect("builder-inited", patch_automodapi)
