.. _recipe:

==========================
Recipes
==========================

A Recipe is a chronological list of Tasks. Every Algorithm object holds
three Recipes — an initialization Recipe, an update Recipe, and a
collect Recipe — and the Dynamics Driver executes them in a fixed
pattern during a simulation. This section describes the three Recipes,
when each one runs, the deep-copy semantics that protect Recipes from
accidental sharing across Algorithm instances, and the idioms used to
edit a Recipe in user code.

For the building-block definition of a Task, see :ref:`Tasks <task>`.
For the State and Parameters dictionaries that Tasks operate on, see
:ref:`State and Parameters Objects <state-and-parameters>`.

----

The three Recipes
=================

An Algorithm object exposes three Recipe attributes:

``algorithm.initialization_recipe``
    Tasks executed once, at the start of every batch, before any time
    step. The initialization Recipe creates the State and Parameters
    entries that the update Recipe and the collect Recipe will read
    and write.

``algorithm.update_recipe``
    Tasks executed once per update time step
    (``sim.settings.dt_update``). The update Recipe is where the
    integrator lives; it propagates the wavefunction and the classical
    coordinate forward in time.

``algorithm.collect_recipe``
    Tasks executed once per collect time step
    (``sim.settings.dt_collect``). The collect Recipe computes the
    quantities to be recorded and copies them into
    ``state["output_dict"]``. After the collect Recipe runs, the
    Dynamics Driver averages ``state["output_dict"]`` across the batch
    into the Data object.

The dynamics core calls
:meth:`Algorithm.execute_recipe <qclab.Algorithm.execute_recipe>` on
each of these in turn:

.. code-block:: python

    # Simplified extract from qclab.dynamics.run_dynamics.
    for sim.t_ind in t_update_iterator:
        if sim.t_ind == 0:
            state, parameters = sim.algorithm.execute_recipe(
                sim, state, parameters, sim.algorithm.initialization_recipe
            )
        if np.mod(sim.t_ind, sim.settings.dt_collect_n) == 0:
            state, parameters = sim.algorithm.execute_recipe(
                sim, state, parameters, sim.algorithm.collect_recipe
            )
            data.add_output_to_data_dict(sim, state, sim.t_ind)
        state, parameters = sim.algorithm.execute_recipe(
            sim, state, parameters, sim.algorithm.update_recipe
        )

The pattern is fixed: the initialization Recipe runs once at
``t_ind == 0``; the collect Recipe runs at every update time step whose
index is a multiple of ``dt_collect_n``; the update Recipe runs at
every update time step.

----

A Recipe is a list of callables
===============================

Each entry of a Recipe is a callable with the Task signature
``f(sim, state, parameters)`` returning ``(state, parameters)``.
:meth:`Algorithm.execute_recipe <qclab.Algorithm.execute_recipe>`
threads ``state`` and ``parameters`` through the list:

.. code-block:: python

    def execute_recipe(self, sim, state, parameters, recipe):
        for func in recipe:
            state, parameters = func(sim, state, parameters)
        return state, parameters

Two consequences follow:

- Tasks must accept the full ``(sim, state, parameters)`` signature.
  A Task that needs extra arguments is wrapped with
  :func:`functools.partial` before being placed in the Recipe (see
  :ref:`The partial idiom <recipe-partial>`).
- The Recipe is **chronological**: Task ``k+1`` is free to read any
  State entry that Task ``k`` has written, and the Recipe writer is
  responsible for ordering. This is true within a single Recipe
  invocation and also across the boundary between consecutive update
  Recipes — anything written into the State object on update step
  ``n`` is still there at the start of update step ``n+1``.

----

Recipe lifetimes within a batch
===============================

The State and Parameters objects are created at the start of every
batch and discarded at the end (see
:ref:`State and Parameters Objects <state-and-parameters>`). The
Recipes themselves do not change between batches — the same Algorithm
object is reused — but the dictionaries they operate on are fresh
every batch.

The implication for Recipe authors is that the initialization Recipe
should treat the State object as if it has just been created with only
``state["seed"]`` set; every other State entry needed by the update
Recipe or the collect Recipe must be created by the initialization
Recipe (or by a subsequent Task in the same Recipe).

----

Deep-copy semantics
===================

A Recipe is stored as a class attribute on the Algorithm class.
:meth:`Algorithm.__init__ <qclab.Algorithm.__init__>` deep-copies
each Recipe into the instance so that mutations on one Algorithm
instance do not leak into another:

.. code-block:: python

    # Inside Algorithm.__init__:
    self.initialization_recipe = copy.deepcopy(self.initialization_recipe)
    self.update_recipe = copy.deepcopy(self.update_recipe)
    self.collect_recipe = copy.deepcopy(self.collect_recipe)

This protects two Algorithm instances of the same class from sharing
state. The cost is that **mutating the class attribute** —
``MeanField.collect_recipe.append(...)`` — has no effect on an
instance that has already been constructed; the instance is holding a
deep copy of whatever was on the class attribute at the moment
``__init__`` ran.

.. warning::

    Do not mutate ``algorithm.collect_recipe`` (or the other Recipes)
    by in-place ``.append`` *before* the Algorithm object is
    instantiated.
    :meth:`Algorithm.__init__ <qclab.Algorithm.__init__>` deep-copies
    the class attribute, so an in-place change made on the class can
    be lost. Use the ``recipe = recipe + [task]`` pattern after
    instantiation instead (see below).

----

Editing a Recipe in user code
=============================

Recipes are ordinary Python lists. The recommended idiom for appending
a Task to an existing Recipe after instantiation is to rebind the
attribute to a new list rather than to call ``.append`` on the
existing list:

.. code-block:: python

    sim.algorithm.collect_recipe = (
        sim.algorithm.collect_recipe + [collect_mean_position]
    )

The reason for the rebinding pattern is consistency: the same
expression works whether the underlying list is the deep-copied
instance attribute or a freshly mutated one, and it makes the change
visible to any reader inspecting the attribute directly.

Inserting a Task at a specific position uses ``list.insert`` after
the Algorithm has been instantiated:

.. code-block:: python

    sim.algorithm.update_recipe.insert(3, my_extra_task)

Removing a Task uses the standard Python list operations. The Tasks
are ordinary function objects, so identity-based removal (``remove``
on the function reference) is unreliable when the Task was inserted
via :func:`functools.partial` — compare on the partial object itself,
or rebuild the list explicitly.

----

.. _recipe-partial:

The partial idiom for ``*_name`` keyword arguments
==================================================

Many built-in Tasks accept ``*_name`` keyword arguments that select
which State or Parameters entry the Task reads or writes. The
standard names are documented in
:ref:`Conventions <conventions>`. To reuse a single Task with
different ``*_name`` arguments, the Tasks are wrapped in
:func:`functools.partial` when placed in a Recipe:

.. code-block:: python

    from functools import partial
    from qclab import tasks

    # Two RK4 sub-step Tasks built from the same underlying Task,
    # bound to different intermediate-coordinate names.
    rk4_step_one = partial(
        tasks.update_z_rk4_k123,
        z_name="z",
        z_k_name="z_1",
        k_name="z_rk4_k1",
    )
    rk4_step_two = partial(
        tasks.update_z_rk4_k123,
        z_name="z",
        z_k_name="z_2",
        k_name="z_rk4_k2",
    )

The built-in mean-field algorithm uses this pattern to assemble the
four RK4 sub-steps from one ``update_z_rk4_k123`` Task and one
``update_z_rk4_k4`` Task, with the intermediate ``z_1``, ``z_2``,
``z_3`` coordinates threaded through by name. See
``src/qclab/algorithms/mean_field.py`` for the full update Recipe.

----

Worked example: reading a built-in Recipe
=========================================

The mean-field algorithm's initialization Recipe is short enough to
read end-to-end:

.. code-block:: python

    initialization_recipe = [
        tasks.initialize_variable_objects,
        tasks.initialize_norm_factor,
        tasks.initialize_z,
        tasks.update_h_q_tot,
    ]

The first Task creates the standard State entries (wavefunction,
classical coordinate, density matrix, output dictionary). The second
Task records the per-batch normalization factor that the Data object
will use when averaging. The third Task draws the initial classical
coordinate from either the Model's ``init_classical`` ingredient or
the default MCMC fallback. The fourth Task evaluates
:math:`\hat{H}_\mathrm{q} + \hat{H}_\mathrm{q\text{-}c}(z)` and stores
it under ``state["h_q_tot"]`` so the first update Recipe can begin
with it already in place.

The collect Recipe is similarly compact:

.. code-block:: python

    collect_recipe = [
        tasks.update_t,
        tasks.update_dm_db_wf,
        tasks.update_quantum_energy_wf,
        tasks.update_classical_energy,
        tasks.collect_t,
        tasks.collect_dm_db,
        tasks.collect_classical_energy,
        tasks.collect_quantum_energy,
    ]

The four ``update_*`` Tasks compute the recordable quantities (time,
diabatic density matrix, quantum energy, classical energy) from the
State object. The four ``collect_*`` Tasks copy each result into
``state["output_dict"]`` under its standard key. After the collect
Recipe returns,
:meth:`Data.add_output_to_data_dict <qclab.data.Data.add_output_to_data_dict>`
averages the entries across the batch axis into ``data.data_dict``.

The update Recipe is longer because it implements the full RK4
integrator. See ``src/qclab/algorithms/mean_field.py`` for the full
listing.

----

When a new Recipe is appropriate
================================

A new Recipe is appropriate when adding a new Algorithm; see
:ref:`Developing Custom Algorithms <developing-algorithms>`. Reusing
the built-in Recipes with extra Tasks appended (the
``mean_position`` example in
:ref:`State and Parameters Objects <state-and-parameters>`) is the
right pattern for recording new quantities or for adding small
side-effects to an existing Algorithm.
