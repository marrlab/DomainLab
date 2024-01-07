"""
run python setup.py install to install DomainLab into system
"""
import os
from setuptools import find_packages, setup

def copy_dir(dir_path="zdata"):
    # root = os.path.dirname(os.path.abspath(__file__))
    root = os.path.normpath(".")
    base_dir = os.path.join(root, "domainlab", dir_path)
    for (dirpath, dirnames, files) in os.walk(base_dir):
        for f in files:
            path = os.path.join(dirpath.split('/', 1)[1], f)
            print(path)
            yield path

setup(
    name='domainlab',
    packages=find_packages(),
    # include_package_data=True,
    # data_files=[
    #               ('bitmaps', ['bm/b1.gif', 'bm/b2.gif']),
    #        ('/opt/local/myproject/etc', ['myproject/config/settings.py', 'myproject/config/other_settings.special']),
    #        ('/opt/local/myproject/etc', [os.path.join('myproject/config', 'cool.huh')]),
    #
    #        ('/opt/local/myproject/etc', [os.path.join('myproject/config', 'other_settings.xml')]),
    #        ('/opt/local/myproject/data', [os.path.join('myproject/words', 'word_set.txt')]),
    #    ],
    # data_files =   [
    #         ('../data', 'data/*')
    #         ],
    # data_files =   [
    #        ('../data', f) for f in copy_dir()
    #        ],
    package_data =   {
                'zdata': [f for f in copy_dir()],
             },
    version='0.3.2',
    description='Library of modular domain generalization for deep learning',
    url='https://github.com/marrlab/DomainLab',
    author='Xudong Sun, et.al.',
    license='MIT',
)
