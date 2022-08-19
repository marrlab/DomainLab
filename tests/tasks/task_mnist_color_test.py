"""
Color MNIST with palette
"""

from domainlab.tasks.task_mnist_color import NodeTaskMNISTColor10
from domainlab.tasks.task_mnist_color import mk_parser_main

def test_fun():
    parser = mk_parser_main()
    args = parser.parse_args(["--te_d", "1", "--dpath", "zout"])
    node = NodeTaskMNISTColor10()
    node.get_list_domains()
    node.list_str_y
    node.init_business(args)
