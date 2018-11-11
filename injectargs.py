def injectArguments(inFunction):
    """
    This function allows to reduce code for initialization of parameters of a method through the decorator(@) notation
    You need to call this function before the method init in this way: @injectArguments
    """
    def outFunction(*args, **kwargs):
        _self = args[0]
        _self.__dict__.update(kwargs)
        # Get all of argument's names of the inFunction
        _total_names = inFunction.func_code.co_varnames[1:inFunction.func_code.co_argcount]
        # Get all of the values
        _values = args[1:]
        # Get only the names that don't belong to kwargs
        _names = [n for n in _total_names if not kwargs.has_key(n)]

        # Match names with values and update __dict__
        _self.__dict__.update(zip(_names,_values))
        return inFunction(*args,**kwargs)

    return outFunction
