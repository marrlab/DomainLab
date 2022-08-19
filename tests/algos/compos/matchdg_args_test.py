import argparse
from domainlab.algos.compos.matchdg_args import add_args2parser_matchdg


def test_fun():
    parser = argparse.ArgumentParser(description='matchdg')
    parser = add_args2parser_matchdg(parser)
    parser.parse_args()
