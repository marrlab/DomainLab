"""Chain of Responsibility"""

__author__ = "Xudong Sun"

import abc
from domainlab.utils.logger import Logger


class Request4Chain(metaclass=abc.ABCMeta):
    """
    define all available fields of request to ensure operation safety
    """
    @abc.abstractmethod
    def convert(self, obj):
        """
        Convert an heavy weight object into request object with pre-defined behavior
        """
        raise NotImplementedError


class AbstractChainNodeHandler(metaclass=abc.ABCMeta):
    """Chain of Responsibility:
    1. Make sure the chain can be constructed successfully even one node fails to
    initialize its designated service/business object so services from other nodes will still
    be available.
    2. To ensure this decoupling, avoid overriding self.__init__() (the initializer/constructor)
    of the handler by using multiple inheritance.
    e.g. inherit both AbstractChainNodeHandler and Factory Interface since doing so couples the
    Factory initializer with the AbstractChainNodeHandler():
    if the initializer/constructor of the Factory does not work, it will
    affect the whole chain.
    3. Instead, return service/business object in a member function of AbstractChainNodeHandler
    self.init_business(), this can result in redundant code but is safest.
    4. A sub-optimal but still acceptable solution is to use Multiple Inheritance (inherit
    AbstractChainNodeHandler and Factory interface) but **only** override the
    self.init_business(*kargs, **kwargs) method (with concrete input arguments)
    of the Chain Handler so the initializer/constructor of the Chain Handler
    will always work. Factory can be returned by calling
    ChainNode.init_business(*kargs, **kwargs).
    This can still be coupling since there might be some interface methods in Factory,
    once you change the parent class, some concrete factories has not implemented that,
    which will break the initalization of the chain.
    """

    def __init__(self, success_node=None):
        """__init__.
        :param success_node: successor chain node which implement the AbstractChainNodeHandler
        interface
        """
        self._success_node = success_node
        self._parent_node = None
        if success_node is not None:
            success_node.set_parent(self)

    def set_parent(self, parent):
        self._parent_node = parent
        parent._success_node = self

    @abc.abstractmethod
    def is_myjob(self, request):
        """is_myjob.
        :param request: subclass can override request object to be string or function
        :return True/False
        """
        raise NotImplementedError

    @abc.abstractmethod
    def init_business(self, *kargs):
        """init_business: initialize **and** return the heavy weight business object for
        doing the real job
        :param request: subclass can override request object to be string or function
        :return: the constructed service object
        """
        raise NotImplementedError

    def handle(self, request):
        """This method invoke self.is_myjob() to check which node in the chain should handle the
        request
        :param request: subclass can override request object to be string or function
        :return: light weight AbstractChainNodeHandler
        """
        if self.is_myjob(request):
            return self
        if self._success_node is not None:
            return self._success_node.handle(request)
        err_msg = str(request) + " does not exist"
        logger = Logger.get_logger()
        logger.info("available options are")
        self.print_options()
        raise NotImplementedError(err_msg)

    def print_options(self):
        logger = Logger.get_logger()
        logger.info(str(self.__class__.__name__))
        if self._parent_node is not None:
            self._parent_node.print_options()


class DummyBusiness():
    message = "dummy business"


class DummyChainNodeHandlerBeaver(AbstractChainNodeHandler):
    """Dummy class to show how to inherit from Chain of Responsibility
    """

    def init_business(self, *kargs, **kwargs):
        return DummyBusiness()

    def is_myjob(self, request):
        """is_myjob.
        :param request: subclass can override request object to be string or function
        :return True/False
        """
        return True


class DummyChainNodeHandlerLazy(AbstractChainNodeHandler):
    """Dummy class to show how to inherit from Chain of Responsibility
    """

    def init_business(self, *kargs, **kwargs):
        return DummyBusiness()

    def is_myjob(self, request):
        """is_myjob.
        :param request: subclass can override request object to be string or function
        :return True/False
        """
        return False
