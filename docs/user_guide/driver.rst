.. _driver:

==========================
Drivers
==========================


QC Lab comes equipped with three dynamics drivers. These are functions that take a Simulation object (see :ref:`Simulations <simulation>`) as input and carry out the dynamics by executing the recipes of the Algorithm object (see :ref:`Algorithms <algorithm>`) associated with the simulation. The three drivers are:

- ``serial_driver``: a serial driver that runs the simulation on a single CPU core,
- ``multiprocessing_driver``: a parallel driver that uses Python's built-in ``multiprocessing`` module to run the simulation on multiple CPU cores,
- ``mpi_driver``: a parallel driver that uses the ``mpi4py`` package to run the simulation on multiple CPU cores, possibly across multiple nodes.

Each driver is responsible for managing the execution of the simulation, including dividing the total number of trajectories into batches (if necessary), distributing the batches across available CPU cores, and collecting the results into a single output Data object. 

All drivers in QC Lab accept the following input arguments (parallel drivers accept an additional argument, see below):

- ``sim``: an instance of the ``qclab.Simulation`` class containing the model, algorithm, and settings for the simulation,
- ``seeds``: an optional array of integers specifying the random seeds for each trajectory in
    the simulation. If not provided, the seeds will be generated automatically.
- ``data``: an input Data object into which the results of the simulation will be added. If not provided, a new Data object will be created.

Generically, a driver is called as:

.. code-block:: python

    data = driver(sim)


Serial Driver
--------------------------

The serial driver runs batches of trajectories sequentially without without requesting a particular set of resources. This means it may use multiple CPU cores if available, but it does not attempt to manage or limit the number of CPU cores used. The serial driver is suitable for running small simulations where parallelization is not necessary, or for debugging purposes. 

.. autofunction:: qclab.dynamics.serial_driver


Parallel Drivers
--------------------------

The parallel drivers use multiple CPU cores to run batches of trajectories concurrently. This can significantly speed up the simulation, especially for large numbers of trajectories. It is important to recognize that the ``sim.settings.batch_size`` now refers to the number of trajectories that will run on a single CPU core at a time. If you have ``N`` CPU cores available and a batch size of ``B``, then up to ``N*B`` trajectories will be simulated concurrently. Having a ``sim.settings.num_trajs = (N+1)*B`` will have an unnecessary overhead since the last ``B`` trajectories will not be able to run concurrently. For that reason it is recommended to set ``sim.settings.num_trajs`` to be a multiple of ``N*B``.

In addition to the arguments described above, the parallel drivers accept ``ntasks``, an integer specifying the number of parallel tasks to use. If ``ntasks`` is not provided, the parallel drivers will attempt to use each available CPU core as a separate task.

.. autofunction:: qclab.dynamics.parallel_driver_multiprocessing

.. autofunction:: qclab.dynamics.parallel_driver_mpi


The MPI driver requires the ``mpi4py`` package to be installed and an MPI execution environment to run. It is suitable for running large simulations on high-performance computing clusters, and can be invoked using the ``mpirun`` or ``mpiexec`` command, as in:

.. code-block:: bash

    mpirun -n 4 python my_simulation_script.py

where ``-n 4`` specifies that the simulation should be run using 4 MPI processes. The mpi driver will automatically distribute the batches of trajectories across the available MPI processes (4 in this case).

An example script that uses the mpi driver can be found in ``examples/mpi_examples/mpi_example.py`` along with a SLURM submission script in the same folder. The full source code of these examples is included here for convenience:

.. dropdown:: mpi_example.py
   :icon: code

   .. literalinclude:: ../../examples/mpi_examples/mpi_example.py
      :language: python
      :linenos:


.. dropdown:: qsubmit.txt
   :icon: code

   .. literalinclude:: ../../examples/mpi_examples/qsubmit.txt
      :language: bash
      :linenos:





Dynamics Core
--------------------------

.. autofunction:: qclab.dynamics.run_dynamics