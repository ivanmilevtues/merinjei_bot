def not_none(arg_name):
    def wrapper(func):
        def inner(self, *args, **kwargs):
            arg = getattr(self, arg_name)
            if arg is None:
                raise TypeError(
                    arg_name + " is not initialized! Use init_" + arg_name + "() or load_"+ arg_name + "() to initialize "
                    + arg_name + " first.")
            return  func(self, *args, **kwargs)
        return inner

    return wrapper