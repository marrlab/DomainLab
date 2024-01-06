"""
args for matchdg
"""


def add_args2parser_matchdg(parser):
    """
    args for matchdg
    """
    # parser = argparse.ArgumentParser()
    parser.add_argument('--tau', type=float, default=0.05,
                        help='factor to magnify cosine similarity')
    parser.add_argument('--epos_per_match_update', type=int, default=5,
                        help='Number of epochs before updating the match tensor')
    parser.add_argument('--epochs_ctr', type=int, default=1,
                        help='Total number of epochs for ctr')
    # args = parser.parse_args("")
    # return args
    return parser
