.. _state-and-parameters:

==================================
State and Parameters Objects
==================================

Every QC Lab task has the same signature::

    def my_task(sim, state, parameters, **kwargs):
        ...
        return state, parameters

The ``state`` and ``parameters`` arguments are dictionaries threaded
through every task in a recipe by
:meth:`Algorithm.execute_recipe <qclab.Algorithm.execute_recipe>`.
Together they carry the per-batch information that the tasks operate on.

.. note::

    The :ref:`Conventions <conventions>` section lists every standard
    key (``wf_db``, ``z``, ``h_q_tot``, and so on) used by the built-in
    tasks. The present section describes how the State object and the
    Parameters object are used and how custom tasks should interact with
    them.

----

Two objects, two lifetimes
==========================

Both dictionaries are created at the start of every batch and discarded
at the end. They are not shared across batches.

The State object holds quantities that are intrinsic to the current
batch of trajectories and that change during the simulation: the
wavefunction, the classical coordinate, eigenvectors, density matrices,
and similar. A task reads named entries from the State object, performs
its computation, and writes named entries back. State-object entries
have a leading batch axis of length ``sim.settings.batch_size``.

The Parameters object holds quantities that are computed once and reused
across multiple update time steps, or quantities that come from outside
the trajectory pipeline. The cached output of an electronic-structure
calculation in the ab initio algorithms is one example.

The dynamics core in :func:`qclab.dynamics.run_dynamics` does not look
at either object directly; it passes them to whichever recipe is being
executed:

.. code-block:: python

    # Inside run_dynamics, each update time step.
    state, parameters = sim.algorithm.execute_recipe(
        sim, state, parameters, sim.algorithm.update_recipe
    )

Because both dictionaries are shared by every task in a recipe, key
collisions matter. Tasks that read or write a quantity should use the
standard key listed in the :ref:`Conventions <conventions>` section
whenever the quantity already has a name there.

----

The ``output_dict`` bridge to the Data object
=============================================

One entry of the State object leaves the trajectory pipeline:
``state["output_dict"]``. Whatever lives there at the end of a collect
time step is averaged across the batch by
:meth:`Data.add_output_to_data_dict <qclab.data.Data.add_output_to_data_dict>`
and stored in the Data object returned by the driver.

Recording a new quantity in QC Lab is therefore a two-step process:

#. An update task computes the quantity and stores the result on the
   State object.
#. A separate collect task copies the quantity into
   ``state["output_dict"]``.

Putting the computation in an update task allows the value to be
available on every update time step (so it can be reused by later
tasks), while the collect task copies it into ``output_dict`` only on
collect time steps.

An example pair that records the mean classical position is:

.. code-block:: python

    import numpy as np
    from qclab.functions import z_to_q

    def update_mean_position(sim, state, parameters,
                             *, z_name="z",
                             mean_position_name="mean_position"):
        """Compute the mean real-space position from the complex coordinate."""
        z = state[z_name]
        m = sim.model.constants.classical_coordinate_mass[np.newaxis, :]
        h = sim.model.constants.classical_coordinate_weight[np.newaxis, :]
        q = z_to_q(z, m, h)
        state[mean_position_name] = q.mean(axis=1)
        return state, parameters

    def collect_mean_position(sim, state, parameters,
                              *, mean_position_name="mean_position",
                              mean_position_output_name="mean_position"):
        """Copy the mean position into output_dict for recording."""
        state["output_dict"][mean_position_output_name] = state[mean_position_name]
        return state, parameters

To wire these into a simulation, append them after the Algorithm object
has been instantiated:

.. code-block:: python

    sim.algorithm.initialization_recipe = (
        sim.algorithm.initialization_recipe + [update_mean_position]
    )
    sim.algorithm.update_recipe = (
        sim.algorithm.update_recipe + [update_mean_position]
    )
    sim.algorithm.collect_recipe = (
        sim.algorithm.collect_recipe + [collect_mean_position]
    )

The update task is included in the initialization recipe so that the
quantity exists at ``t = 0`` before the first collect time step runs.
After the simulation, the new key is available in
``data.data_dict["mean_position"]``.

.. warning::

    Do not modify ``algorithm.collect_recipe`` (or the other recipes) by
    in-place ``.append`` before the Algorithm object is instantiated.
    :meth:`Algorithm.__init__ <qclab.Algorithm.__init__>` deep-copies the
    class attribute, so an in-place change made on the class can be
    lost. Use the ``recipe = recipe + [task]`` pattern after
    instantiation instead.

----

The ``*_name`` keyword convention
=================================

A task should not hard-code keys for the State object or the Parameters
object. Each key that a task reads or writes is exposed as a ``*_name``
keyword argument with a sensible default. Recipes can then rebind the
same task to different keys using ``functools.partial``:

.. code-block:: python

    from functools import partial
    from qclab import tasks

    # The same RK4 sub-step, rebound to use z_2 instead of z_1.
    rk4_step_two = partial(
        tasks.update_z_rk4_k123,
        z_name="z",
        z_k_name="z_2",
        k_name="z_rk4_k2",
    )

The built-in mean-field and FSSH algorithms use this pattern to reuse a
single set of RK4 sub-step tasks across the four sub-steps of the
integrator. See ``src/qclab/algorithms/mean_field.py`` for an example.

When writing a new task, follow the same rule: every State-object or
Parameters-object key the task reads or writes is a ``*_name`` keyword
argument with a default that matches the standard names in the
:ref:`Conventions <conventions>` section.

----

Task docstring sections
=======================

The Sphinx configuration adds a set of Napoleon sections used by QC Lab
tasks to document their inputs and outputs. The sections recognized by
the built-in tasks are:

``Optional Keyword Arguments``
    The ``*_name`` keyword arguments accepted by the task and their
    defaults.

``Reads``
    Entries of the State object (or the Parameters object) that the
    task expects to already exist. Include the shape and dtype.

``Writes``
    Entries of the State object (or the Parameters object) that the
    task creates or overwrites. Include the shape and dtype.

``Constants and Settings``
    Entries of ``sim.model.constants`` or ``sim.settings`` that the task
    consults.

``Ingredients``
    Standard slot names that the task calls via ``sim.model.get(...)``.

``Shapes and dtypes``
    Free-form notes about shape conventions when they are not obvious
    from Reads / Writes.

A representative collect-task docstring looks like:

.. code-block:: rst

    Optional Keyword Arguments
    --------------------------
    t_name:
        Name of the time variable in the State object.
    t_output_name:
        Name of the time variable in the output dictionary.

    Reads
    -----
    state[t_name]: ndarray of shape (B,), dtype=float64
        Time in each trajectory.

    Writes
    ------
    state["output_dict"][t_output_name]: ndarray of shape (B,), dtype=float64
        Time in each trajectory.

    Notes
    -----
    * B = sim.settings.batch_size
