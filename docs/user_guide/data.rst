.. _data:

==========================
Data
==========================

Data Objects
---------------------------

Data objects are instances of the ``qclab.data.Data`` class and are used to store and manage the results of a simulation. They provide methods for collecting, processing, and saving data, as well as logging errors or warnings during a simulation.

In general, a Data object has the following attributes:

- ``data_dict``: a dictionary that stores the results of the simulation. Each key in the dictionary corresponds to a specific quantity that was collected during the simulation, and the value is an array containing the values of that quantity averaged over the trajectories.
- ``log``: a string that stores the log of errors or warnings that occurred during the simulation.

Data objects provide several methods for managing and processing the data they contain, including:

- ``add_data``: adds data from an existing Data object to the current one.
- ``save``: saves the Data object to a file.
- ``load``: loads a Data object from a file (this adds to any existing data).

These methods are documented here:

.. autofunction:: qclab.data.Data.add_data
.. autofunction:: qclab.data.Data.save
.. autofunction:: qclab.data.Data.load


Example
---------------------------

Here is a simple example of running a simulation and plotting from the Data object returned by the driver:

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

    t = data.data_dict['t']
    plt.plot(t, np.real(np.einsum('tii->ti',data.data_dict['dm_db'])))
    plt.title('Diabatic populations')
    plt.show()