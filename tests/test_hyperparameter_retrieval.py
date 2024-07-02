"""
unit test for hyperparameter parsing
"""
import pytest
from domainlab.arg_parser import mk_parser_main
from domainlab.utils.hyperparameter_retrieval import get_gamma_reg

def test_store_dict_key_pair_single_value():
    """Test to parse a single gamma_reg parameter"""
    parser = mk_parser_main()
    args = parser.parse_args(['--gamma_reg', '0.5'])
    assert args.gamma_reg == 0.5

def test_store_dict_key_pair_dict_value():
    """Test to parse a dict for the gamma_reg"""
    parser = mk_parser_main()
    args = parser.parse_args(['--gamma_reg', 'dann=1.0,jigen=2.0'])
    assert args.gamma_reg == {'dann': 1.0, 'jigen': 2.0}

def test_get_gamma_reg_single_value():
    """Test to retrieve a single gamma_reg parameter which is applied to all objects"""
    parser = mk_parser_main()
    args = parser.parse_args(['--gamma_reg', '0.5'])
    assert get_gamma_reg(args, 'dann') == 0.5

def test_get_gamma_reg_dict_value():
    """Test to retrieve a dict of gamma_reg parameters for different objects"""
    parser = mk_parser_main()
    args = parser.parse_args(['--gamma_reg', 'default=5.0,dann=1.0,jigen=2.0'])
    assert get_gamma_reg(args, 'dann') == 1.0
    assert get_gamma_reg(args, 'jigen') == 2.0
    assert get_gamma_reg(args, 'nonexistent') == 5.0  # if we implement other
    # model/trainers,
    # since not specified in command line arguments, the new model/trainer
    # called "nonexistent" should
    # get the default value 5.0.

def test_exception():
    """Test to not specify a default value"""
    parser = mk_parser_main()
    args = parser.parse_args(['--gamma_reg', 'dann=1.0'])

    with pytest.raises(ValueError, match="If a gamma_reg dict is specified"):
        get_gamma_reg(args, 'jigen')
