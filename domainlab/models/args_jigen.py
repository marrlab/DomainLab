"""
hyper-parameters for JiGen
"""
def add_args2parser_jigen(parser):
    """
    hyper-parameters for JiGen
    """
    parser.add_argument('--nperm', type=int, default=31,
                        help='number of permutations')
    parser.add_argument('--pperm', type=float, default=0.1,
                        help='probability of permutating the tiles \
                        of an image')
    parser.add_argument('--jigen_ppath', type=str, default=None,
                        help='npy file path to load numpy array with each row being \
                        permutation index, if not None, nperm and grid_len has to agree \
                        with the number of row and columns of the input array')
    parser.add_argument('--grid_len', type=int, default=3,
                        help='length of image in tile unit')
    return parser
