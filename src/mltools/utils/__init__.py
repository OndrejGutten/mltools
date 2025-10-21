import os
import importlib

# Get the current directory
current_dir = os.path.dirname(__file__)

# List all .py files except __init__.py
module_files = [
    f for f in os.listdir(current_dir)
    if f.endswith('.py') and f != '__init__.py'
]

for module_file in module_files:
    module_name = module_file[:-3]
    module = importlib.import_module(f'.{module_name}', package=__name__)
    for attr in dir(module):
        if not attr.startswith('_'):
            globals()[attr] = getattr(module, attr)