import datetime

import swiift.__about__

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "SWIIFT"
copyright = f"2023--{datetime.date.today().year}, Nicolas Mokus"

author = "Nicolas Mokus"
version = swiift.__about__.__version__
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "sphinx.ext.extlinks",
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
# Prevent duplicates, are __init__ are constructed by attrs
napoleon_include_init_with_doc = False

autoapi_dirs = ["../../src/swiift"]
autoapi_type = "python"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]

extlinks = {
    "doi": ("https://dx.doi.org/%s", "DOI: %s"),
}

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
