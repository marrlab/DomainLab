from domainlab.compos.pcr.p_chain_handler import (  DummyChainNodeHandlerBeaver, 
                                                    DummyChainNodeHandlerLazy
                                                    )


def test_chain_of_responsibility():
    """
    todo: https://stackoverflow.com/a/223586/3390810
    """
    chain = DummyChainNodeHandlerBeaver(None)
    chain = DummyChainNodeHandlerLazy(chain)
    node = chain.handle(None)
    business = node.init_business()
    business.message
