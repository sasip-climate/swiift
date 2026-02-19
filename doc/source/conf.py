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
    "sphinx.ext.autodoc",
    "autoapi.extension",
    "sphinxcontrib.bibtex",
    "sphinx.ext.intersphinx",
    "myst_nb",
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
# Prevent duplicates, as __init__ are constructed by attrs
napoleon_include_init_with_doc = False

autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_typehints_format = "short"
autoapi_dirs = ["../../src/swiift"]
autoapi_type = "python"
autoapi_options = [
    "members",
    # "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_python_class_content = "class"


def skip_member_filter(app, what, name, obj, skip, options):
    # Specify the members documented multiple times due to import to
    # packages. The behaviour we seek is to only document members at
    # their highest level of accessibility.
    to_skip = (
        "swiift.api.Experiment",
        "swiift.api.api.Experiment",
        "swiift.api.load_pickle",
        "swiift.api.api.load_pickle",
        "swiift.api.load_pickles",
        "swiift.api.api.load_pickles",
        "swiift.model.Ocean",
        "swiift.model.model.Ocean",
    )
    if name in to_skip:
        return True
    return None


def setup(sphinx):
    sphinx.connect("autoapi-skip-member", skip_member_filter)


bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

# Line included at the beginning of every rst file
rst_prolog = f"""
.. |project| replace:: {project}
"""

myst_enable_extensions = ["colon_fence"]
nb_custom_formats = {
    ".md": ["jupytext.reads", {"fmt": "mystnb"}],
}
nb_execution_mode = "off"
nb_number_source_lines = True
nb_render_markdown_format = "myst"
