import subprocess
import warnings
from subprocess import CalledProcessError

from domainlab.utils.logger import Logger


def get_git_tag(print_diff=False):
    flag_not_commited = False
    logger = Logger.get_logger()
    try:
        subprocess.check_output(
            ['git', 'diff-index', '--quiet', 'HEAD'])
    except CalledProcessError:
        logger.warning("\n\n")
        logger.warning("!!!: not committed yet")
        warnings.warn("!!!: not committed yet")
        flag_not_commited = True
        logger.warning("\n\n")
        try:
            diff_byte = subprocess.check_output(['git', 'diff'])
            if print_diff:
                logger.info(str(diff_byte))  # print is currently ugly, do not use!
        except Exception:
            logger = Logger.get_logger()
            logger.warning("not in a git repository")
            warnings.warn("not in a git repository")
    try:
        tag_byte = subprocess.check_output(
            ["git", "describe", "--always"]).strip()
        logger.info(str(tag_byte))
        tag_str = str(tag_byte)
        git_str = tag_str.replace("'", "")
        if flag_not_commited:
            git_str = git_str + "_not_commited"
        return git_str
    except Exception:
        logger = Logger.get_logger()
        logger.warning("not in a git repository")
        warnings.warn("not in a git repository")
    return "no_git_version"
