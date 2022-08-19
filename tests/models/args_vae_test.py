import argparse
from domainlab.models.args_vae import add_args2parser_vae

def test_fun():
    parser = argparse.ArgumentParser(description='diva')
    parser = add_args2parser_vae(parser)
    parser.parse_args()
