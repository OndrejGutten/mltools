import importlib
import os
import inspect
from pathlib import Path

# Get the directory of the current file
current_dir = Path(__file__).parent

# Iterate through all .py files in the directory (excluding __init__.py)
for file in current_dir.glob("*.py"):
    module_name = file.stem
    if module_name != "__init__":
        # Skip import until this part is finished
        if module_name == "preprocess":
            continue
        # Import the module
        module = importlib.import_module(f".{module_name}", package=__name__)
        
        # Get all classes in the module
        for name, obj in inspect.getmembers(module, lambda m: inspect.isclass(m) or inspect.isfunction(m)):
            # Check if the class is defined in this module (to avoid imports from other modules)
            if obj.__module__ == module.__name__:
                # Add the class to the module's globals so it can be accessed directly
                globals()[name] = obj