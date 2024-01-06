def add_args2parser_vae(parser):
    parser.add_argument('--zd_dim', type=int, default=64,
            help='diva: size of latent space for domain')
    parser.add_argument('--zx_dim', type=int, default=0,
            help='diva: size of latent space for unobserved')
    parser.add_argument('--zy_dim', type=int, default=64,
            help='diva, hduva: size of latent space for class')
    #  HDUVA
    parser.add_argument('--topic_dim', type=int, default=3,
            help='hduva: number of topics')

    parser.add_argument('--nname_encoder_x2topic_h',
            type=str, default=None,
            help='hduva: network from image to topic distribution')

    parser.add_argument('--npath_encoder_x2topic_h',
                        type=str, default=None,
                        help='hduva: network from image to topic distribution')

    parser.add_argument('--nname_encoder_sandwich_x2h4zd',
                        type=str, default=None,
                        help='hduva: network from image and topic to zd')
    parser.add_argument('--npath_encoder_sandwich_x2h4zd',
                        type=str, default=None,
                        help='hduva: network from image and topic to zd')

    # ERM, ELBO
    parser.add_argument('--gamma_y', type=float, default=None,
            help='diva, hduva: multiplier for y classifier')
    parser.add_argument('--gamma_d', type=float, default=None,
            help='diva: multiplier for d classifier from zd')



    # Beta VAE part
    parser.add_argument('--beta_t', type=float, default=1.,
            help='hduva: multiplier for KL topic')
    parser.add_argument('--beta_d', type=float, default=1.,
            help='diva: multiplier for KL d')
    parser.add_argument('--beta_x', type=float, default=1.,
            help='diva: multiplier for KL x')
    parser.add_argument('--beta_y', type=float, default=1.,
            help='diva, hduva: multiplier for KL y')
    return parser
