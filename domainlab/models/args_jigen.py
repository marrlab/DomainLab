def add_args2parser_jigen(parser):
    parser.add_argument('--nperm', type=int, default=31,
                        help='number of permutations')
    parser.add_argument('--pperm', type=float, default=0.7,
                        help='probability of permutating the tiles \
                        of an image')
    parser.add_argument('--grid_len', type=int, default=3,
                        help='length of image in tile unit')
    return parser
