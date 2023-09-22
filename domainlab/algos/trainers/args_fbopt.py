"""
feedback opt
"""


def add_args2parser_fbopt(parser):
    """
    append hyper-parameters to the main argparser
    """

    parser.add_argument('--k_i_gain', type=float, default=0.001,
                        help='PID control gain for integrator')

    parser.add_argument('--mu_clip', type=float, default=1e4,
                        help='maximum value of mu')

    parser.add_argument('--coeff_ma', type=float, default=0.5,
                        help='exponential moving average')

    parser.add_argument('--exp_shoulder_clip', type=float, default=10,
                        help='clip before exponential operation')

    parser.add_argument('--ini_setpoint_ratio', type=float, default=0.99,
                        help='before training start, evaluate reg loss, \
                        setpoint will be 0.9 of this loss')

    # the following hyperparamters do not need to be tuned
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
    parser.add_argument('--myoptic_pareto', action='store_true', default=False,
                        help='update theta^{k} upon pareto descent')
    return parser
