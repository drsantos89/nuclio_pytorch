# Install [Conda](https://docs.conda.io/en/latest/)

I recommend installing Conda [Miniforge](https://github.com/conda-forge/miniforge) to handle virtual environments.
```bash
cd ~/Downloads
```
```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh
```
```bash
bash Miniforge3-MacOSX-x86_64.sh
```

# Create a virtual enviroment and activate it
Conda environments are isolated environments where packages and dependencies can be installed and managed separately from the system-wide packages. This allows for different versions of packages and dependencies to be used in different environments, without interfering with each other.

```bash
conda create --name py10 python=3.10
```
```bash
conda activate py10
```

# [Pre-commit](https://pre-commit.com/)
A framework for managing and maintaining multi-language pre-commit hooks.

## Instalation
```bash
conda install pre-commit
```
```bash
pre-commit install
```

## [pre-commit hooks](https://pre-commit.com/hooks.html)
Git hook scripts are useful for identifying simple issues before submission to code review.

- check-added-large-files <br>
makes sure no large, probably data files, are added to git <br>
- check-json
- check-toml
- check-yaml <br>
checks for a valid format of these three configuration files
- requirements-txt-fixer <br>
orders the requirements alphabetically
- trailing-whitespace <br>
removes white spaces after the end of the line
- id: end-of-file-fixer <br>
adds an empty line to the end of the file
- id: no-commit-to-branch <br>
args: [--branch, master, --branch, main]

## [isort](https://pycqa.github.io/isort/)
isort is a Python utility / library to sort imports alphabetically, and automatically separated into sections and by type.

- profile = "black" <br>
makes sure that inplace changes compatible with black
- line_length = 88 <br>
allows up to 88 characters per line

## [black](https://black.readthedocs.io/en/stable/)
Black is a PEP 8 compliant opinionated formatter with its own style.
- line-length = 88 <br>
allows up to 88 characters per line

## [flake8](https://flake8.pycqa.org/en/latest/)
A Tool For Style Guide Enforcement <br>
- docstring-convention = numpy <br>
- max-line-length = 88 <br>
- select = B, C, E, F, W, B9, ISC <br>
- ignore = E203, E402, E501, E722, B001, W503
## [mypy](https://mypy.readthedocs.io/en/stable/index.html)
Mypy is a static type checker for Python.

- ignore_missing_imports = true
- no_implicit_optional = true
- check_untyped_defs = true
- strict_equality = true
- warn_redundant_casts = true
- warn_unused_ignores = true
- show_error_codes = true
- disallow_any_generics = true
- disallow_incomplete_defs = true
- disallow_untyped_defs = true

## [bandit](https://bandit.readthedocs.io/en/latest/)
Bandit is a tool designed to find common security issues in Python code.

# [DVC](https://dvc.org/doc)
DVC allows you to track, manage, and version your data and models in a similar way to how Git tracks code.

## Installation
```bash
conda install dvc
```

## Initialization
```bash
dvc init
dvc remote add <remote-path>
```
## add files or folder to dvc
```bash
dvc add dvc/<local-path>
```
## push files to remote server
```bash
dvc push
```

# GitHub Actions
A continuous integration and continuous delivery (CI/CD) platform that allows you to automate your build, test, and deployment pipeline.

The GitHub Actions workflow located at github/workflows/ci.yaml is triggered on pull requests.

It contains two jobs, "Lint" and "Test", that run on the latest version of Ubuntu.

The "Lint" job sets up Python 3.10, and then installs the dependencies needed for linting. It then runs various linters such as isort, black, flake8, mypy and bandit to check for code formatting, style and security issues.

The "Test" job also sets up Python 3.10, installs dependencies, including dev dependencies, and runs pytest to execute the test cases.

This GitHub CI workflow is used for checking the code quality, formatting and security before merging the pull request.
