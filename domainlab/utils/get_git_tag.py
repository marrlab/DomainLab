import subprocess
import warnings
from subprocess import CalledProcessError


def get_git_tag(print_diff=False):
    flag_not_commited = False
    try:
        subprocess.check_output(
            ['git', 'diff-index', '--quiet', 'HEAD'])
    except CalledProcessError:
        print("\n\n")
        warnings.warn("!!!: not committed yet")
        flag_not_commited = True
        print("\n\n")
        try:
            diff_byte = subprocess.check_output(['git', 'diff'])
            if print_diff:
                print(diff_byte)  # print is currently ugly, do not use!
        except Exception:
            warnings.warn("not in a git repository")
    try:
        tag_byte = subprocess.check_output(
            ["git", "describe", "--always"]).strip()
        print(tag_byte)
        tag_str = str(tag_byte)
        git_str = tag_str.replace("'", "")
        if flag_not_commited:
            git_str = git_str + "_not_commited"
        return git_str
    except Exception:
        warnings.warn("not in a git repository")
    return "no_git_version"
