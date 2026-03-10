.. _install:

====================
Installing QC Lab
====================

This guide walks you through installing QC Lab from source or from PyPI, using pip.

Requirements
------------
- Python 3.8 or newer.
- pip (Python package installer).
- git (optional, for cloning the repository directly).


Installing from PyPI
--------------------
QC Lab can be installed from the Python Package Index (PyPI) by executing


.. code-block:: bash

      pip install qclab


To install QC Lab without h5py or Numba support, execute

.. code-block:: bash

      pip install qclab --no-deps
      pip install numpy tqdm

The second line installs the remaining required dependencies manually.


Installing from source
----------------------

QC Lab can be installed from source by downloading the `latest release <https://github.com/tempelaar-team/qclab/releases>`_,
unpacking it, and executing

.. code-block:: bash

      pip install ./

from inside its topmost directory (where the `pyproject.toml` file is located).

.. note::

      QC Lab doesn’t enforce third-party dependencies. If you encounter resolver conflicts or install errors, the quickest fix is to install in a clean Python environment (via `venv` or `conda`). Alternatively, reconcile package versions in your existing environment until the requirements are satisfied.


That’s it! QC Lab should now be installed and ready for use.
