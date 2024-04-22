# Assume imports and any necessary setup are already done
import argparse
from domainlab.arg_parser import StoreDictKeyPair, mk_parser_main
from domainlab.utils.hyperparameter_retrieval import get_gamma_reg

def test_store_dict_key_pair_single_value():
    parser = mk_parser_main()
    args = parser.parse_args(['--gamma_reg', '0.5'])
    assert args.gamma_reg == 0.5

def test_store_dict_key_pair_dict_value():
    parser = mk_parser_main()
    args = parser.parse_args(['--gamma_reg', 'dann=1.0,diva=2.0'])
    assert args.gamma_reg == {'dann': 1.0, 'diva': 2.0}

def test_get_gamma_reg_single_value():
    parser = mk_parser_main()
    args = parser.parse_args(['--gamma_reg', '0.5'])
    assert get_gamma_reg(args, 'dann') == 0.5

def test_get_gamma_reg_dict_value():
    parser = mk_parser_main()
    args = parser.parse_args(['--gamma_reg', 'default=5.0,dann=1.0,diva=2.0'])
    print(args)
    assert get_gamma_reg(args, 'dann') == 1.0
    assert get_gamma_reg(args, 'diva') == 2.0
    assert get_gamma_reg(args, 'nonexistent') == 5.0
