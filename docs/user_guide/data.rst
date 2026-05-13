.. _data:

==========================
Data
==========================

Data Objects
---------------------------

Data objects are instances of the ``qclab.data.Data`` class and are
used to store and manage the results of a simulation. They provide
methods for collecting, processing, and saving data, and they capture
log output produced during a simulation.

A Data object has the following attributes:

- ``data_dict``: a dictionary that stores the results of the
  simulation. Each key corresponds to a quantity collected during the
  simulation, and the value is an array containing the trajectory-
  averaged values of that quantity. Two bookkeeping keys are always
  present: ``"seed"`` (the array of trajectory seeds that contributed
  to the average) and ``"norm_factor"`` (the denominator used by
  ``add_data`` to perform a trajectory-count-weighted merge).
- ``log``: a string that stores log messages emitted during the
  simulation. The drivers populate this attribute by reading from the
  in-memory log stream configured by
  :func:`qclab.utils.configure_memory_logger` once the simulation has
  finished.

Data objects provide methods for managing and processing the data they
contain.

- ``add_output_to_data_dict``: the hook called by
  :func:`qclab.dynamics.run_dynamics` at every collect time step. It
  merges ``state["output_dict"]`` into ``data_dict``, broadcasting new
  keys to the full ``(n_collect, ...)`` time axis on the first
  occurrence and dividing the per-batch sum by the ``norm_factor``.
- ``add_data``: merges another Data object into this one using a
  trajectory-count-weighted average of every collected quantity. The
  drivers call this method to combine batches, and users can call it to
  stitch multiple runs of the same simulation together.
- ``save``: writes ``data_dict`` and ``log`` to disk. Uses HDF5 (via
  ``h5py``) when available; falls back to ``numpy.savez`` when
  ``h5py`` is not installed (``qclab.utils.DISABLE_H5PY``).
- ``load``: reads a Data file from disk into the current Data object.
  The loaded data is merged in via ``add_data``, so loading on top of a
  non-empty Data object accumulates trajectories rather than
  overwriting them.

These methods are documented here:

.. autofunction:: qclab.data.Data.add_output_to_data_dict
.. autofunction:: qclab.data.Data.add_data
.. autofunction:: qclab.data.Data.save
.. autofunction:: qclab.data.Data.load

.. note::

    ``add_data`` performs a weighted merge: for every collected key
    other than ``"seed"`` and ``"norm_factor"``, the merged value is
    ``(d1 * n1 + d2 * n2) / (n1 + n2)``, where ``n1`` and ``n2`` are
    the ``norm_factor`` values of the two Data objects (typically the
    number of trajectories that contributed). This makes it possible
    to stitch together runs with different batch sizes. Manually
    editing ``norm_factor`` on a Data object will produce an incorrect
    weighted average; in such a case it is preferable to construct a
    fresh Data object.

Saving and loading
~~~~~~~~~~~~~~~~~~

A simulation's output can be saved to disk and reloaded later:

.. code-block:: python

    data = serial_driver(sim)
    data.save("results.h5")          # HDF5 if h5py is installed,
                                     # otherwise .npz with savez

    from qclab import Data
    reloaded = Data().load("results.h5")

Loading an existing Data file on top of a non-empty Data object merges
the two via ``add_data``. This is useful when accumulating large
simulations from periodic checkpoint files:

.. code-block:: python

    big = Data()
    for chunk in checkpoint_files:
        big.load(chunk)
    # ``big`` now contains all chunks averaged together.


Example
---------------------------

Here is a simple example of running a simulation and plotting from the
Data object returned by the driver:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from qclab import Simulation
    from qclab.models import SpinBoson
    from qclab.algorithms import MeanField
    from qclab.dynamics import serial_driver

    sim = Simulation()
    sim.model = SpinBoson()
    sim.algorithm = MeanField()
    sim.initial_state["wf_db"] = np.array([1, 0], dtype=complex)
    data = serial_driver(sim)

    t = data.data_dict["t"]
    populations = np.real(np.einsum("tii->ti", data.data_dict["dm_db"]))
    plt.plot(t, populations)
    plt.title("Diabatic populations")
    plt.show()

    # Simulation log captured during the run is available on data.log.
    print(data.log)