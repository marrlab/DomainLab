"""
feedback opt
"""


def add_args2parser_fbopt(parser):
    """
    append hyper-parameters to the main argparser
    """

    parser.add_argument(
        "--k_i_gain", type=float, default=0.001,
        help="PID control gain for integrator, if k_i_gain_ratio is not None, \
        then this value will be overwriten"
    )

    parser.add_argument(
        "--k_i_gain_ratio",
        type=float,
        default=None,
        help="set k_i_gain to be ratio of \
                        initial saturation k_i_gain",
    )

    parser.add_argument(
        "--mu_clip", type=float, default=1e4, help="maximum value of mu"
    )

    parser.add_argument(
        "--mu_min", type=float, default=1e-6, help="minimum value of mu"
    )

    parser.add_argument(
        "--mu_init", type=float, default=0.001,
        help="initial value for each component of the multiplier vector"
    )

    parser.add_argument(
        "--coeff_ma", type=float, default=0.5,
        help="exponential moving average"
    )

    parser.add_argument(
        "--coeff_ma_output_state",
        type=float,
        default=0.1,
        help="output (reguarization loss) exponential moving average",
    )

    parser.add_argument(
        "--coeff_ma_setpoint",
        type=float,
        default=0.9,
        help="setpoint average (coeff for previous setpoint)",
    )

    parser.add_argument(
        "--exp_shoulder_clip",
        type=float,
        default=5,
        help="clip delta(control error): \
        R(reg-loss)-b(setpoint) before exponential operation: \
        exp[clip(R-b, exp_shoulder_clip)].\
        exponential magnifies control error, so this argument \
        defines the maximum rate of change of multipliers \
        exp(5)=148",
    )

    parser.add_argument(
        "--ini_setpoint_ratio",
        type=float,
        default=0.99,
        help="before training start, evaluate reg loss, \
                        setpoint will be 0.9 of this loss",
    )

    parser.add_argument(
        "--force_feedforward",
        action="store_true",
        default=False,
        help="use feedforward scheduler",
    )

    parser.add_argument(
        "--force_setpoint_change_once",
        action="store_true",
        default=False,
        help="continue trainiing until the setpoint changed at least once: \
              up to maximum epos specified",
    )

    parser.add_argument(
        "--no_tensorboard",
        action="store_true",
        default=False,
        help="disable tensorboard",
    )

    parser.add_argument(
        "--no_setpoint_update",
        action="store_true",
        default=False,
        help="disable setpoint update",
    )

    parser.add_argument(
        "--tr_with_init_mu",
        action="store_true",
        default=False,
        help="disable setpoint update",
    )

    # FIXME: change arguments from str to boolean
    parser.add_argument(
        "--overshoot_rewind",
        type=str,
        default="yes",
        help="overshoot_rewind, for benchmark, use yes or no",
    )

    parser.add_argument(
        "--setpoint_rewind",
        type=str,
        default="no",
        help="rewind setpoint, for benchmark, use yes or no",
    )

    parser.add_argument(
        "--str_diva_multiplier_type",
        type=str,
        default="gammad_recon",
        help="which penalty to tune, only useful to DIVA model",
    )

    return parser
