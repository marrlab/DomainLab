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

    parser.add_argument('--mu_min', type=float, default=1e-6,
                        help='minimum value of mu')

    parser.add_argument('--mu_init', type=float, default=0.001,
                        help='initial beta for multiplication')

    parser.add_argument('--coeff_ma', type=float, default=0.5,
                        help='exponential moving average')

    parser.add_argument('--coeff_ma_output_state', type=float, default=0.9,
                        help='state exponential moving average of reguarlization loss')

    parser.add_argument('--coeff_ma_setpoint', type=float, default=0.9,
                        help='setpoint average coeff for previous setpoint')

    parser.add_argument('--exp_shoulder_clip', type=float, default=5,
                        help='clip before exponential operation')

    parser.add_argument('--ini_setpoint_ratio', type=float, default=0.99,
                        help='before training start, evaluate reg loss, \
                        setpoint will be 0.9 of this loss')

    parser.add_argument('--no_tensorboard', action='store_true', default=False,
                        help='disable tensorboard')

    parser.add_argument('--no_setpoint_update', action='store_true', default=False,
                        help='disable setpoint update')

    parser.add_argument('--overshoot_rewind', type=str, default="yes",
                        help='overshoot_rewind, for benchmark, use yes or no')

    parser.add_argument('--setpoint_rewind', type=str, default="no",
                        help='setpoing_rewind, for benchmark, use yes or no')

    parser.add_argument('--str_diva_multiplier_type', type=str, default="gammad_recon",
                        help='which penalty to tune')

    return parser
