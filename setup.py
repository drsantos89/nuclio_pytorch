"""Setup file for <templateds>."""
from setuptools import find_packages, setup

VERSION = "0.0.0"

DESCRIPTION = "My template for data science projects"

setup(
    name="templateds",
    version=VERSION,
    description=DESCRIPTION,
    author="Diogo Santos",
    author_email="drsantos989@gmail.com",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[],
    extras_require={
        "dev": [
            "isort",
            "black[jupyter]",
            "flake8",
            "flake8-bugbear",
            "flake8-builtins",
            "flake8-comprehensions",
            "flake8-docstrings",
            "mypy",
            "bandit[toml]",
            "pytest",
            "pytest-cov",
            "pre-commit",
        ],
    },
)
