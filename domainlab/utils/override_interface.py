def override_interface(interface_class):
    """overrides.
    :param interface_class:  the interface class name, always specify this
    explicitly as otherwise interface_class is going to be the nearest
    function it decorate, and argument "method2override" of returned
    function "overrider" accept will be the current child class

    .. code-block:: python

       class BaseClass()
       class Child(BaseClass):
           @overrides(BaseClass)
           def fun(self):
               pass
    """
    def overrider(method2override):
        """overrider.

        :param method:  the method to be decorated, method signature will be
        returned intactly
        """
        assert method2override.__name__ in dir(interface_class)
        return method2override

    return overrider
