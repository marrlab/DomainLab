# Installation of DomainLab

## Create a virtual environment for DomainLab (strongly recommended)

`conda create --name domainlab_py39 python=3.9`

then

`conda activate domainlab_py39`

### Install Development version via github

Suppose you have cloned the repository and have changed directory to the cloned repository.

```norun
pip install -r requirements.txt
```
then

`python setup.py install`

#### Dependencies management
-   [python-poetry](https://python-poetry.org/) and use the configuration file `pyproject.toml` in this repository.

###  Install Release
It is strongly recommended to create a virtual environment first, then
- Install via `pip install domainlab`
