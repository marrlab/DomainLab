from importlib.metadata import entry_points

from domainlab.cli import domainlab_cli


def test_entry_point():
    eps = entry_points()
    cli_entry = eps.select(group='console_scripts')['domainlab']
    assert cli_entry.load() == domainlab_cli

def test_domainlab_cli(monkeypatch):
    test_args = ['--te_d', '1', '2', '--tr_d', '0', '3', '--task', 'mnistcolor10', '--epos', '500', '--bs', '16', '--model', 'erm', '--nname', 'conv_bn_pool_2', '--lr', '1e-3']
    monkeypatch.setattr('sys.argv', ['domainlab'] + test_args)
    domainlab_cli()
