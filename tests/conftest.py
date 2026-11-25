"""Test-session configuration.

JAX is only used on host to trace/lower Pallas kernels; it must never open ``/dev/vfio/0``,
which TorchTPU owns exclusively. Pin JAX to CPU before any test imports it.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
