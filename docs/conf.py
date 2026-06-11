import os
import sys
from datetime import datetime
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

repository_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repository_root))

with (repository_root / "pyproject.toml").open("rb") as pyproject_file:
    release = tomllib.load(pyproject_file)["project"]["version"]

project = "cpp_oti_lib"
author = "cpp_oti_lib contributors"
copyright = f"{datetime.now().year}, {author}"

extensions = [
    "sphinx.ext.autosectionlabel",
    "breathe",
]

autosectionlabel_prefix_document = True
templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "generated",
    "api/xml",
    "api/html",
    "cpp_oti_lib_documentation.tex",
]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "cpp_oti_lib"
html_show_sourcelink = False

breathe_projects = {
    "cpp_oti_lib": os.path.abspath("api/xml"),
}
breathe_default_project = "cpp_oti_lib"
breathe_domain_by_extension = {
    "hpp": "cpp",
}
