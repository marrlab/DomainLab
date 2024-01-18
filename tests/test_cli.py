"""
test 'domainlab' entry point
"""

from importlib.metadata import entry_points

from domainlab.cli import domainlab_cli


def test_entry_point():
    """
    Test the entry point for the 'domainlab' console script.

    This function retrieves all entry points and asserts that the 'domainlab'
    entry point is correctly associated with the 'domainlab_cli' function.
    """
    eps = entry_points()
    cli_entry = eps.select(group="console_scripts")["domainlab"]
    assert cli_entry.load() is domainlab_cli


def test_domainlab_cli(monkeypatch):
    """
    Test the 'domainlab_cli' function by simulating command-line arguments.

    This function uses the 'monkeypatch' fixture to set the command-line
    arguments for the 'domainlab_cli' function and then calls it to ensure
    it processes the arguments correctly. The test arguments simulate a
    representative command-line input for the 'domainlab' tool.
    """
    test_args = [
        "--te_d",
        "1",
        "2",
        "--tr_d",
        "0",
        "3",
        "--task",
        "mnistcolor10",
        "--epos",
        "500",
        "--bs",
        "16",
        "--model",
        "erm",
        "--nname",
        "conv_bn_pool_2",
        "--lr",
        "1e-3",
    ]
    monkeypatch.setattr("sys.argv", ["domainlab"] + test_args)
    domainlab_cli()
