def add_args2parser_matchdg(parser):
    # parser = argparse.ArgumentParser()
    parser.add_argument('--tau', type=int, default=0.05,
                        help='factor to magnify cosine similarity')
    parser.add_argument('--epos_per_match_update', type=int, default=5,
                        help='Number of epochs before updating the match tensor')
    parser.add_argument('--epochs_ctr', type=int, default=None,
                        help='Total number of epochs for ctr')
    parser.add_argument('--epochs_erm', type=int, default=None,
                        help='Total number of epochs for erm')
    parser.add_argument('--penalty_ws', type=float, default=0.1,
                        help='Penalty weight for Matching Loss')
    # args = parser.parse_args("")
    # return args
    return parser
