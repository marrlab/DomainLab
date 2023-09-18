"""
feedback opt
"""


def add_args2parser_fbopt(parser):
    """
    append hyper-parameters to the main argparser
    """
    parser.add_argument('--init_mu4beta', type=float, default=0.001,
                        help='initial beta for multiplication')
    parser.add_argument('--beta_mu', type=float, default=1.1,
                        help='how much to multiply mu each time')
    parser.add_argument('--delta_mu', type=float, default=None,
                        help='how much to increment mu each time')
    parser.add_argument('--budget_mu_per_step', type=int, default=5,
                        help='number of mu iterations to try')
    parser.add_argument('--budget_theta_update_per_mu', type=int, default=5,
                        help='number of theta update for each fixed mu')
    parser.add_argument('--anchor_bar', action='store_true', default=False,
                        help='use theta bar as anchor point')
    return parser
