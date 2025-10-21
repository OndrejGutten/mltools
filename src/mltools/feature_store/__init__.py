import os
import importlib
import pkgutil
import inspect

__all__ = []  # we'll populate

def _export_core_symbols(module):
    for name, obj in inspect.getmembers(module):
        if (inspect.isfunction(obj) or inspect.isclass(obj)) and not name.startswith("_"):
            # only export things defined in that module
            if getattr(obj, "__module__", None) == module.__name__:
                globals()[name] = obj
                __all__.append(name)

# --- core: import everything except interface ---
core_dir = os.path.join(os.path.dirname(__file__), "core")
for _, modname, ispkg in pkgutil.iter_modules([core_dir]):
    if ispkg or modname == "interface":
        continue
    mod = importlib.import_module(f".core.{modname}", __name__)
    _export_core_symbols(mod)

# expose core.interface as a submodule
interface = importlib.import_module(".core.interface", __name__)
globals()["interface"] = interface
__all__.append("interface")

# --- utils: attach the package ONLY (no re-export of its members) ---
utils = importlib.import_module(".utils", __name__)
globals()["utils"] = utils
__all__.append("utils")
