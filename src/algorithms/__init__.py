"""\
  Init   module.

Classes:
- None

Functions:
- None

Created: 2025-05-03
"""

import types, sys, importlib, pathlib
_src_pkg = importlib.import_module('src.algorithms')
# expose all names
globals().update(_src_pkg.__dict__)
# Make submodules importable as algorithms.<sub>
for _name, _mod in sys.modules.items():
    if _name.startswith('src.algorithms') and _name != 'src.algorithms':
        sys.modules[_name.replace('src.', '')] = _mod
# Ensure pkg resources
__path__ = [str(pathlib.Path(_src_pkg.__file__).parent)]
