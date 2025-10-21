# submodule/utils/__init__.py
from .utils import *  # re-export functions/classes defined in utils.py

# Optional but nice for linters/type checkers:
try:
    # if utils.py defines __all__, use it
    from .utils import __all__  # noqa: F401
except Exception:
    # otherwise build a default __all__
    import inspect as _inspect
    __all__ = [
        n for n, o in globals().items()
        if not n.startswith("_") and not _inspect.ismodule(o)
    ]
    del _inspect
