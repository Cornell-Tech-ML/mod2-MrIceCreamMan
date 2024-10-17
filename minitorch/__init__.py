"""Initialize the package and expose key classes and functions.

This file serves as the package initializer and handles the importing and exposing of
key modules, classes, and functions to make them available at the package level.

Imports:
--------
- `MathTest`, `MathTestVariable` from `testing` module:
  Import specific classes for testing mathematical functions.

- All public members from `module`, `testing`, and `datasets` modules:
  Import all public functions, classes, and variables from these modules to simplify
  access and usage at the package level.

Notes
-----
- `# type: ignore # noqa: F401,F403`:
  These comments are used to suppress type-checking warnings (`type: ignore`) and flake8
  warnings:
  - `F401`: Ignore "imported but unused" warnings.
  - `F403`: Ignore "from module import *" usage warnings.

"""

from .testing import MathTest, MathTestVariable  # type: ignore # noqa: F401,F403
from .tensor_data import *  # noqa: F401,F403
from .tensor import *  # noqa: F401,F403
from .tensor_ops import *  # noqa: F401,F403
from .tensor_functions import *  # noqa: F401,F403
from .datasets import *  # noqa: F401,F403
from .optim import *  # noqa: F401,F403
from .testing import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
from .autodiff import *  # noqa: F401,F403
from .scalar import *  # noqa: F401,F403
from .scalar_functions import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
