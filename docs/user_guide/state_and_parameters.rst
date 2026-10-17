.. _state-and-parameters:

====================
State and Parameters
====================

The Dynamics Driver creates two dictionaries at the outset of each batch of trajectories: the State object and the Parameters object. Both are ordinary Python dictionaries whose entries are keyed by name. Tasks operate on the State object and, where the Algorithm calls for it, on the Parameters object (see :ref:`Tasks <task>`). Their roles differ, and the sections below describe each in turn.


The State Object
----------------

The State object, generically denoted ``state``, holds the trajectory-resolved dynamical variables: the wavefunction, the complex-valued classical coordinate, and the other quantities that evolve as the simulation proceeds. Its array entries carry a leading batch dimension ``B = sim.settings.batch_size``, so the diabatic wavefunction ``state["wf_db"]`` has shape ``(B, num_quantum_states)`` and the complex-valued classical coordinate ``state["z"]`` has shape ``(B, num_classical_coordinates)``. Working over this dimension lets a single Task advance the whole batch at once.

The Dynamics Driver seeds the State object with one entry, the per-trajectory random ``seed``. Every other entry is created by an initialization Task. The ``initialize_variable_objects`` Task promotes each array you place in ``sim.initial_state`` into a batched State entry: an array of shape ``original_shape`` becomes a State entry of shape ``(B, *original_shape)``, with the original array copied into each trajectory slice. This is how the diabatic wavefunction you assign to ``sim.initial_state["wf_db"]`` (see :ref:`Simulations <simulation>`) reaches the dynamics. Entries whose names begin with an underscore are held private and are left out of this promotion.

Several State entries are generic to any quantum–classical dynamics simulation in QC Lab:

- ``wf_db`` / ``wf_adb`` — the diabatic / adiabatic wavefunction (whichever applies), shape ``(B, num_quantum_states)``.
- ``z`` — the complex-valued classical coordinate, shape ``(B, num_classical_coordinates)``.
- ``seed`` — the per-trajectory random seed, shape ``(B,)``.
- ``norm_factor`` — a scalar equal to the batch size, used to normalize summed output to a trajectory average.

A particular Algorithm adds whatever further entries it needs. Surface-hopping Algorithms, for instance, add ``act_surf_ind``, the active-surface index of each trajectory.


The Output Dictionary
~~~~~~~~~~~~~~~~~~~~~~~

Alongside the promoted variables, ``initialize_variable_objects`` creates ``state["output_dict"]``, an initially empty dictionary that the collect Recipe fills with the quantities to be recorded. A collect Task writes a batch-resolved array into it, as in ``state["output_dict"]["dm_db"] = state["dm_db"]``. On each collect time step, the Dynamics Driver passes ``state["output_dict"]`` to the Data object, which averages the recorded quantities over the batch and stores them. For further details, see :ref:`Data Objects <data>`.


The Parameters Object
---------------------

The Parameters object, generically denoted ``parameters``, carries quantities that change as the simulation proceeds so that the Model can depend on them (for instance, a Hamiltonian that varies in time).

Every Ingredient receives the Parameters object but not the State object (see :ref:`Ingredients <ingredient>`). So when a quantity that varies during the simulation must reach an Ingredient, the Parameters object is the route: a Task writes the quantity into it, and an Ingredient reads that value later, when called.

The Dynamics Driver creates the Parameters object empty, and whether it comes to hold anything depends on the Algorithm; many leave it empty.


Lifecycle
---------

The State and Parameters objects are created fresh for each batch of trajectories and discarded once the batch finishes; the Data object carries results between batches. Within a batch, the sequence is:

#. Before the dynamics start, the initialization Recipe runs once, populating the State object — and the Parameters object, where the Algorithm calls for it.
#. On each update time step, the update Recipe advances the State object's evolving entries, foremost the wavefunction and the complex-valued classical coordinate — and, where the Algorithm maintains a time-varying quantity in the Parameters object, updates that as well.
#. On each collect time step, the collect Recipe writes the recorded quantities into ``state["output_dict"]``, which the Data object averages and stores.

Nothing written to the State or Parameters object persists across batches. The Dynamics Driver runs this batch-by-batch sequence; see :ref:`Drivers <driver>`.
