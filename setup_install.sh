# source distribution
python setup.py sdist
pip install -e .
#!/bin/bash
python setup.py develop
echo "================================="
python setup.py install --record installed_files.txt

cat installed_files.txt
xargs rm -rf < install_files.txt

echo "===========wheel======================"

python setup.py bdist_wheel
