from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.abspath("../"))
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "QC Lab"
year = datetime.now().year
copyright = f"2025-{year}, Tempelaar Team"
author = "Tempelaar Team"
# html_title = "QC Lab Documentation"
# html_short_title = "QC Lab Docs"
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "sphinx_togglebutton",
    "sphinxcontrib.mermaid",
    "sphinx.ext.graphviz",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]





autosummary_generate = True  # build autosummary pages
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
autodoc_typehints_description_target = "documented_params"
napoleon_custom_sections = [
    ("Requires", "params_style"),
    ("Optional Keyword Arguments", "params_style"),
    ("Constants and Settings", "params_style"),
    ("Ingredients", "params_style"),
    ("Reads", "params_style"),
    ("Writes", "params_style"),
    ("Shapes and dtypes", "params_style"),
    ("Optional Keyword Arguments", "params_style"),
]


# Map your favorite short type names to canonical refs (optional)
napoleon_type_aliases = {
    "ndarray": "numpy.ndarray",
    "ArrayLike": "numpy.typing.ArrayLike",
    "complex128": "numpy.complex128",
}


# Optional: link out to NumPy docs for types
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3", None),
}





autodoc_typehints = "description"  # "signature" "both"

# graphviz_output_format = "svg"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# import sphinx_rtd_theme
# html_theme = "sphinx_rtd_theme"
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme = "pydata_sphinx_theme"  # "bizstyle"#
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    # "show_toc_level": 2,   # show h2/h3 (your subsections)
    # "secondary_sidebar_items": [],  # remove all right-sidebar widgets
    "body_max_width": "100%",  # Remove maximum width constraint
    "github_url": "https://github.com/tempelaar-team/qclab",
    "collapse_navigation": True,
    "show_nav_level": 2,
    # "secondary_sidebar_items": ["page-toc", "sourcelink"],
    "external_links_new_tab": True,
    "logo": {
        "logo_only": True,
        "image_light": "_static/images/logo-light.png",
        "image_dark": "_static/images/logo-dark.png",
    },
    "show_toc_level": 3,  # show h2/h3 in right sidebar
}

html_sidebars = {
    "**": ["sidebar-nav-bs.html", "sidebar-ethical-ads.html"],
    # "software_reference/ingredients/ingredients": [],
    # or for a whole folder
    "interactive_docs/index": [],
    # "interactive_docs/spin-boson-example/*": [],
} 

# Remove the right sidebar (“On this page”, buttons, etc.)
# html_theme_options = {
#     "secondary_sidebar_items": [],  # remove all right-sidebar widgets
# }

