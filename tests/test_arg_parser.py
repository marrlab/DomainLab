'''
Code coverage issues:
    https://app.codecov.io/gh/marrlab/DomainLab/blob/master/domainlab/arg_parser.py
    - lines 144-150
'''

from domainlab.arg_parser import parse_cmd_args, mk_parser_main

def test_arg_parser():
    args = parse_cmd_args()
    print(args.lr)
