from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# The manual describes the intended stable release, including during candidacy.
release = "1.0.0"

project = "BCMP Py"
author = ""
version = release

latex_documents = [
    ("index", "bcmppy.tex", r"BCMP\textsuperscript{Py}", author, "manual"),
]
latex_elements = {
    "classoptions": ",oneside",
    "preamble": r"""
\makeatletter
\let\cleardoublepage\clearpage
\makeatother
""",
}

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "vignette/index.rst"]
default_role = "py:obj"
nitpicky = True
nitpick_ignore_regex = [
    (
        "py:class",
        r"(anndata\.AnnData|numpy\.ndarray|logging\.Logger|pathlib\.Path|sequence|bcmp\.models\..*)",
    ),
]

html_theme = "pydata_sphinx_theme"
html_title = "BCMP Python"
html_favicon = "_static/bcmp-favicon-32.png"
html_static_path = ["_static"]
html_css_files = ["bcmp.css"]
html_show_sphinx = False
html_show_sourcelink = False
html_copy_source = False
html_sidebars = {"**": []}
html_context = {
    "bcmp_r_docs_url": "https://guan-pujun.github.io/bcmp-r/",
}
html_theme_options = {
    "navbar_start": ["bcmp-navbar-logo"],
    "navbar_align": "left",
    "navigation_with_keys": True,
    "show_toc_level": 2,
    "logo": {
        "image_light": "_static/bcmp-logo-light.svg",
        "image_dark": "_static/bcmp-logo-dark.svg",
        "alt_text": "BCMP Py documentation home",
        "text": "Python",
    },
    "secondary_sidebar_items": {
        "**": ["bcmp-package-badge", "page-toc"],
        "index": [],
    },
}

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

sphinx_gallery_conf = {
    "examples_dirs": str(ROOT / "vignettes"),
    "gallery_dirs": "vignette",
    "filename_pattern": r"plot_vignette\.py$",
    "download_all_examples": False,
    "remove_config_comments": True,
    "show_memory": False,
    "abort_on_example_error": True,
    "image_scrapers": ("matplotlib",),
    "thumbnail_size": (480, 320),
    "within_subsection_order": "FileNameSortKey",
}


def _strip_gallery_extras(app, docname: str, source: list[str]) -> None:
    """Keep gallery pages focused on the rendered vignette."""
    if not docname.startswith("vignette/"):
        return

    download_note_marker = "\n.. only:: html\n\n    .. note::\n        :class: sphx-glr-download-link-note\n"
    title_marker = "\n.. rst-class:: sphx-glr-example-title\n"
    if download_note_marker in source[0] and title_marker in source[0]:
        before, remainder = source[0].split(download_note_marker, maxsplit=1)
        _, after = remainder.split(title_marker, maxsplit=1)
        source[0] = before + title_marker + after

    timing_marker = "\n\n.. rst-class:: sphx-glr-timing\n"
    if timing_marker in source[0]:
        source[0] = source[0].split(timing_marker, maxsplit=1)[0]

    signature_marker = "\n\n.. only:: html\n\n .. rst-class:: sphx-glr-signature\n"
    if signature_marker in source[0]:
        source[0] = source[0].split(signature_marker, maxsplit=1)[0]


def _align_index_browser_title(app, pagename, templatename, context, doctree) -> None:
    """Keep the home-page browser title concise and package-specific."""
    if pagename == "index":
        context["title"] = "BCMP Python"
        context["docstitle"] = ""


def setup(app) -> None:
    app.connect("source-read", _strip_gallery_extras)
    app.connect("html-page-context", _align_index_browser_title)
