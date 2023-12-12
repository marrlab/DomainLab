"""
run python setup.py install to install DomainLab into system
"""
import os
from setuptools import find_packages, setup

def copy_dir():
    root = os.path.dirname(os.path.abspath(__file__))
    dir_path = 'domainlab/data'
    base_dir = os.path.join(root, dir_path)
    for (dirpath, dirnames, files) in os.walk(base_dir):
        for f in files:
            path = os.path.join(dirpath.split('/', 1)[1], f)
            print(path)
            yield path

setup(
    name='domainlab',
    packages=find_packages(),
    #include_package_data=True,
    #data_files =   [
    #        ('../data', 'data/*')
    #        ],
    # data_files =   [
    #        ('../data', f) for f in copy_dir()
    #        ],
    package_data =   {
            'data': [f for f in copy_dir()],
            },
    version='0.1.9',
    description='Library of Domain Generalization',
    url='https://github.com/marrlab/DomainLab',
    author='Xudong Sun, et.al.',
    license='MIT',
)
