''' Decorators for different callbacks.'''

import functools

def on_finish(callbacks=[], cache={}, obj=None):
    ''' General on function finish callback decorator,
    i.e., this callback activates after a function has been running.
    '''
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                for callback in callbacks:
                    callback(obj, result, cache)
        return wrapper
    return decorator



def __init__(self, callbacks):
    self.callbacks = callbacks
    self.cache: Dict[str, Any] = {}

@final
@on_finish(callbacks=self.callbacks, cache=self.cache, obj=self)
def inner_learning(self, adapt: bool = True) -> Dict[str, Any]:
    ...