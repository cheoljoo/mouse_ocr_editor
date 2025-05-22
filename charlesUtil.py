import sys
import inspect

def getClassFunctionName():
    """
        get class and function name of function
    """
    frame = inspect.currentframe().f_back
    func_name = frame.f_code.co_name
    class_name = ''

    self_obj = frame.f_locals.get('self')
    if self_obj:
        class_name = self_obj.__class__.__name__

    return class_name + '::' + func_name
