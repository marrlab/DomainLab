"""
Code coverage issues:
    https://app.codecov.io/gh/marrlab/DomainLab/blob/master/domainlab/utils/get_git_tag.py
    - lines 10-20
    - lines 28, 30-32
"""
from domainlab.utils.get_git_tag import get_git_tag


def test_git_tag():
    """
    test git_tag
    """
    get_git_tag(print_diff=True)


def test_git_tag_error():
    """
    test git_tag error
    """
    # add one line to the file
    with open("data/ztest_files/dummy_file.py", "a") as f:
        f.write("\n# I am a dummy command")
    get_git_tag(print_diff=True)
    # delete the last line on the file again
    with open("data/ztest_files/dummy_file.py", "r") as f:
        lines = f.readlines()
        lines = lines[:-1]
    with open("data/ztest_files/dummy_file.py", "w") as f:
        for num, line in enumerate(lines):
            if num == len(lines) - 1:
                f.write(line[:-2])
            else:
                f.write(line)
