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
        then this value will be overwriten, see doc for k_i_gain_ratio"
    )

    parser.add_argument(
        "--k_i_gain_ratio",
        type=float,
        default=None,
        help="set k_i_gain to be ratio of initial saturation k_i_gain \
        which K_I * delta = exp_shoulder_clip(saturation value), solve \
        for K_I, where delta = reg loss - setpoint. \
        Now independent of the scale of delta, the K_I gain will be set so \
        that the multiplier will be increased at a rate defined by \
        exp_shoulder_clip",
    )

    parser.add_argument(
        "--mu_clip", type=float, default=1e4,
        help="maximum value of mu: mu_clip should be large enough so that the \
        regularization loss as penalty can be weighed superior enough to \
        decrease."
    )

    parser.add_argument(
        "--mu_min", type=float, default=1e-6, help="minimum value of mu, mu \
        can not be negative"
    )

    parser.add_argument(
        "--mu_init", type=float, default=0.001,
        help="initial value for each component of the multiplier vector"
    )

    parser.add_argument(
        "--coeff_ma", type=float, default=0.5,
        help="exponential moving average of delta \
        (reg minus setpoint as control error): \
        move_ave=move_ave + coeff*delta(current value)"
    )

    parser.add_argument(
        "--coeff_ma_output_state",
        type=float,
        default=0.1,
        help="output (reguarization loss) exponential moving average \
        move_ave=move_ave*coeef + reg(current value)",
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
        exp(5)=148, exp_shoulder_clip should not be too big, \
        if exp_shoulder_clip is small, then more like exterior point method",
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
        help="continue training until the setpoint changed at least once: \
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

    parser.add_argument(
        "--no_overshoot_rewind",
        action="store_true",
        default=False,
        help="disable overshoot rewind: when reg loss satisfies setpoint \
        already, then set activation=K_I*delta = 0",
    )

    parser.add_argument(
        "--setpoint_rewind",
        action="store_true",
        default=False,
        help="rewind setpoint",
    )

    # this arg is only used when model is set to be "diva"
    parser.add_argument(
        "--str_setpoint_ada",
        type=str,
        default="DominateAllComponent()",
        help="which setpoint adaptation criteria to use",
    )

    # this arg is only used when model is set to be "diva"
    parser.add_argument(
        "--str_diva_multiplier_type",
        type=str,
        default="gammad_recon",
        help="which penalty to tune, only useful to DIVA model",
    )

    return parser
