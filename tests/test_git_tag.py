'''
Code coverage issues:
    https://app.codecov.io/gh/marrlab/DomainLab/blob/master/domainlab/utils/get_git_tag.py
    - lines 10-20
    - lines 28, 30-32
'''
from domainlab.utils.get_git_tag import get_git_tag

def test_git_tag():
    """
    test git_tag
    """
    get_git_tag(print_diff=True)
