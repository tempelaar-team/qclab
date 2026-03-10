.. _simulation:

===========================
Simulations
===========================

Simulations in QC Lab are carried out by instances of the ``qclab.Simulation`` class. These Simulation objects contain the model and Algorithm objects that define the system to be simulated and the method of simulation, respectively. The Simulation object also contains an instance of the ``qclab.Constants`` class (``sim.settings``) which defines the settings of the simulation, such as the time step, total simulation time, and number of trajectories to be simulated. The Simulation object also contains a dictionary (``sim.initial_state``) in which the initial state of the system is defined (these are typically algorithm specific).

Simulation Objects
---------------------------

A Simulation object, generically denoted ``sim``, is an instance of the ``qclab.Simulation`` class. It contains the following attributes:

- ``sim.model``: The Model object that defines the system to be simulated.
- ``sim.algorithm``: The Algorithm object that defines the method of simulation.
- ``sim.settings``: An instance of the ``qclab.Constants`` class that defines the settings of the simulation.
- ``sim.initial_state``: A dictionary that defines the initial state of the system.

A Simulation object containing a default mean-field simulation of the spin-boson model can be created as:

.. code-block:: python

    from qclab import Simulation
    from qclab.models import SpinBoson
    from qclab.algorithms import MeanField

    sim = Simulation()
    sim.model = SpinBoson()
    sim.algorithm = MeanField()
    sim.initial_state = np.zeros((sim.model.constants.num_quantum_states,), dtype=complex)
    sim.initial_state[0] = 1.0 + 0.0j 

where ``input_settings`` is a dictionary of settings that overwrite the default settings of the simulation described below.

Simulation Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A default simulation (i.e. one provided with no input dictionary of settings) has the following settings:

- ``tmax``: The total simulation time (default: ``10.0``).
- ``dt_update``: The update time step of the simulation (default: ``0.001``).
- ``dt_collect``: The collect time step of the simulation (default: ``0.1``).
- ``num_trajs``: The total number of trajectories to be simulated (default: ``100``).
- ``batch_size``: The number of trajectories to be simulated at a time (default: ``25``).
- ``progress_bar``: Whether to display a progress bar during the simulation (default: ``True``).
- ``debug``: Whether to run the simulation in debug mode (default: ``False``).

These settings can be changed by passing a dictionary of settings to the simulation constructor, as in:

.. code-block:: python

    input_settings = {
        "tmax": 5.0,
        "dt_update": 0.01,
        "num_trajs": 200
    }
    sim = Simulation(settings=input_settings)

This will create a simulation with a total time of ``5.0``, an update time step of ``0.01``, and ``200`` trajectories, while all other settings will take their default values.

Alternatively, a simulation's settings can be changed after the simulation has been created by modifying the attributes of the ``sim.settings`` object directly, as in:

.. code-block:: python

    sim.settings.tmax = 5.0
    sim.settings.dt_update = 0.01
    sim.settings.num_trajs = 200


Running a Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once a Simulation object has been created and populated with a model, algorithm, and initial state, the simulation can be run by passing the simulation to a dynamics driver. See :ref:`Dynamics Drivers <driver>` for more information.

.. code-block:: python

    from qclab.dynamics import serial_driver

    data = serial_driver(sim)

The resulting Data object is a dictionary containing the outputs of the simulation, which are defined by the collect tasks of the algorithm (see :ref:`Algorithms <algorithm>`). For more information on the Data objects see :ref:`Data Objects <data>`.
