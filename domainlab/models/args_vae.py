def add_args2parser_vae(parser):
    parser.add_argument('--zd_dim', type=int, default=64,
                        help='size of latent space for domain')
    parser.add_argument('--zx_dim', type=int, default=0,
                        help='size of latent space for unobserved')
    parser.add_argument('--zy_dim', type=int, default=64,
                        help='size of latent space for class')
    #  HDUVA
    parser.add_argument('--topic_dim', type=int, default=3,
                        help='dim latent space for topic')

    parser.add_argument('--topic_h_dim', type=int, default=8,
                        help='dim latent space for topic')

    parser.add_argument('--img_h_dim', type=int, default=8,
                        help='dim latent space for topic')

    parser.add_argument('--nname_topic_distrib_img2topic',
                        type=str, default=None,
                        help='network from image to topic distribution')
    parser.add_argument('--npath_topic_distrib_img2topic',
                        type=str, default=None,
                        help='network from image to topic distribution')

    parser.add_argument('--nname_encoder_sandwich_layer_img2h4zd',
                        type=str, default=None,
                        help='network from image to topic distribution')
    parser.add_argument('--npath_encoder_sandwich_layer_img2h4zd',
                        type=str, default=None,
                        help='network from image to topic distribution')

    # ERM, ELBO
    parser.add_argument('--gamma_y', type=float, default=None,
                        help='multiplier for y classifier')
    parser.add_argument('--gamma_d', type=float, default=None,
                        help='multiplier for d classifier from zd')



    # Beta VAE part
    parser.add_argument('--beta_t', type=float, default=1.,
                        help='multiplier for KL topic')
    parser.add_argument('--beta_d', type=float, default=1.,
                        help='multiplier for KL d')
    parser.add_argument('--beta_x', type=float, default=1.,
                        help='multiplier for KL x')
    parser.add_argument('--beta_y', type=float, default=1.,
                        help='multiplier for KL y')
    return parser
