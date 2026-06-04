.. _task:

==========================
Tasks
==========================

Tasks are functions that carry out the elementary operations of an Algorithm object by modifying the State object and optionally the Parameters object, which are both Python dictionaries.
Built-in Tasks can be found in the ``qclab.tasks`` module and are documented in this section.

A generic Task has the form:

.. code-block:: python

    def my_task(sim, state, parameters, **kwargs):
        # Get any keyword arguments, with default values if not provided.
        kwarg1 = kwargs.get('kwarg1', default_value1)
        # Carry out the task by modifying state and parameters
        return state, parameters

where ``sim`` is an instance of the Simulation class (see :ref:`Simulations <simulation>`), and ``**kwargs`` are any additional keyword arguments that customize the Task to the context in which it is used. Generally, keyword arguments specify
which elements of the State and Parameters objects are to be used and/or modified by the Task (e.g., the name of the wavefunction or Hamiltonian attributes).


If a Task is used with keyword arguments, they must be provided when the Task is included in the Algorithm object's Recipe by using the ``partial`` function from the ``functools`` module, as in:

.. code-block:: python

    from functools import partial
    # Specifying keyword arguments when including a Task in a Recipe.
    algorithm.initialization_recipe = [partial(my_task, kwarg1=value1), ...]
    # Including a Task without keyword arguments.
    algorithm.initialization_recipe = [another_task, ...]


Vectorization
--------------------------

In QC Lab, attributes of the State and Parameters objects are vectorized by default, meaning that they have an additional leading dimension that indexes multiple trajectories.
For example, the diabatic wavefunction attribute ``wf_db`` of the State object has shape ``(sim.settings.batch_size, sim.model.constants.num_quantum_states)``, where ``sim.settings.batch_size`` is the number of trajectories in the batch and ``sim.model.constants.num_quantum_states`` is the number of quantum states in the model.


Initialization, Update, and Collect Tasks
------------------------------------------------

Tasks are organized into three categories: initialization Tasks, update Tasks, and collect Tasks.
Initialization Tasks create objects in the State and Parameters objects that are needed for the simulation, update Tasks propagate the attributes of the State and Parameters objects forward in time, and collect Tasks gather and process the results of the simulation into the output dictionary.

Examples of these tasks are:

.. code-block:: python

    def my_initialization_task(sim, state, parameters, **kwargs):
        # Create an attribute in the State object.
        shape = (sim.settings.num_trajs, sim.model.constants.num_quantum_states)
        state["new_attribute"] = np.zeros(shape, dtype=complex)
        return state, parameters

    def my_update_task(sim, state, parameters, **kwargs):
        # Update an attribute in the State object.
        shape = (sim.settings.num_trajs, sim.model.constants.num_quantum_states)
        state["new_attribute"] = state["new_attribute"] + 1j*np.ones(shape, dtype=complex)
        return state, parameters

    def my_collect_task(sim, state, parameters, **kwargs):
        # Collect results into the output dictionary.
        state["output_dict"]["new_attribute"] = state["new_attribute"]
        return state, parameters

These Tasks can then be included in the appropriate Recipe of an Algorithm object. Notice that none of these Tasks have keyword arguments and so can be included directly in Recipes without using ``partial``.

Built-in Tasks
--------------------------
Built-in Tasks can be found in the ``qclab.tasks`` module and are documented below.

.. note::

   All Tasks assume that the Model object has a minimal set of constants including ``num_quantum_states`` (the number of quantum states), ``num_classical_coordinates`` (the number of classical coordinates), ``classical_coordinate_mass`` (the mass of the classical coordinates), and ``classical_coordinate_weight`` (the weight of the classical coordinates). These constants are discussed in :ref:`Models <model>`. For brevity we exclude explicit mention of these constants in the Task documentation.


Initialization Tasks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: qclab.tasks.initialization_tasks
   :members:
   :undoc-members:
   :member-order: bysource
   :exclude-members: __all__, __doc__, __annotations__


Update Tasks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: qclab.tasks.update_tasks
   :members:
   :undoc-members:
   :member-order: bysource
   :exclude-members: __all__, __doc__, __annotations__


Collect Tasks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: qclab.tasks.collect_tasks
   :members:
   :undoc-members:
   :member-order: bysource
   :exclude-members: __all__, __doc__, __annotations__