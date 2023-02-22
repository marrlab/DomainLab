"""
domain invariant adversarial trainer hyper-parmaeters
"""


def add_args2parser_dial(parser):
    """
    append hyper-parameters to the main argparser
    """
    parser.add_argument('--dial_steps_perturb', type=int, default=3,
                        help='how many gradient step to go to find an image as adversarials')
    parser.add_argument('--dial_noise_scale', type=float, default=0.001,
                        help='variance of gaussian noise to inject on pure image')
    parser.add_argument('--dial_lr', type=float, default=0.003,
                        help='learning rate to generate adversarial images')
    parser.add_argument('--dial_epsilon', type=float, default=0.031,
                        help='pixel wise threshold to perturb images')
    return parser
