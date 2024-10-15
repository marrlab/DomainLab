"""
miro trainer configurations
"""


def add_args2parser_miro(parser):
    """
    append hyper-parameters to the main argparser
    """
    arg_group_miro = parser.add_argument_group("miro")
    arg_group_miro.add_argument(
        "--layers2extract_feats",
        nargs="*",
        default=None,
        help="layer names separated by space to extract features",
    )
    return parser
