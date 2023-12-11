"""
run python setup.py install to install DomainLab into system
"""
from setuptools import find_packages, setup
setup(
    name='domainlab',
    packages=find_packages(),
    version='0.1.7',
    description='Library of Domain Generalization',
    author='Xudong Sun, et.al.',
    license='MIT',
)
