.. _algorithm:

==========================
Algorithms
==========================

Algorithms in QC Lab define the sequence of operations that evolve the system defined by the Model object (see :ref:`Models <model>`) in time.
They are composed of three recipes which define initialization steps, update steps, and collect steps that together define the desired algorithm.
Each recipe is a list of "tasks" (see :ref:`Tasks <task>`) which are executed in the order specified by the recipe list. 
Algorithms define the transient quantities of an algorithm in the State object, which is an instance of a dictionary.


Algorithms in QC Lab can are tailored to Model objects defined in adiabatic or diabatic bases (see :ref:`Models <model>`) in order to optimize their
performance. Such tailoring breaks the compatibility between an algorithm implemented assuming a diabatic basis and those models implemented without 
such a basis (and vice versa). As an example, the `AbInitioFewestSwitchesSurfaceHopping` and `AbInitioMeanField` algorithms can only be used with models defined 
in an adiabatic basis. In most cases, model problems are defined in a diabatic basis and so we tailor the present adiabatic algorithms towards ab initio simulations
which are the most common use case for an adiabatic basis.


.. _algorithm_objects:

Algorithm Objects
-----------------------

Algorithm objects in QC Lab are instances of the ``qclab.Algorithm`` class. Each Algorithm object is composed of three recipes: an initialization recipe ``algorithm.initialization_recipe``, an update recipe ``algorithm.update_recipe``, and a collect recipe ``algorithm.collect_recipe``. Like a Model object, an Algorithm object has an instance of the Constants class ``algorithm.settings`` which contains the settings specific to the algorithm. Unlike the Model object, Algorithm objects do not have internal constants and so there is no initialization method as there is for Model objects (see :ref:`Models <model>`). Instead, the settings of the Algorithm object are set directly by the user during or after instantiation of the Algorithm object.

The empty Algorithm class is:


.. code-block:: python


    class Algorithm:
        """
        Algorithm class for defining and executing algorithm recipes.
        """

        def __init__(self, default_settings=None, settings=None):
            if settings is None:
                settings = {}
            if default_settings is None:
                default_settings = {}
            # Merge default settings with user-provided settings.
            settings = {**default_settings, **settings}
            # Construct a Constants object to hold settings.
            self.settings = Constants()
            # Put settings from the dictionary into the Constants object.
            for key, val in settings.items():
                setattr(self.settings, key, val)
            # Copy the recipes and output variables to ensure they are not shared
            # across instances.
            self.initialization_recipe = copy.deepcopy(self.initialization_recipe)
            self.update_recipe = copy.deepcopy(self.update_recipe)
            self.collect_recipe = copy.deepcopy(self.collect_recipe)

        initialization_recipe = []
        update_recipe = []
        collect_recipe = []

        def execute_recipe(self, sim, state, parameters, recipe):
            """
            Carry out the given recipe for the simulation by running
            each task in the recipe.
            """
            for func in recipe:
                state, parameters = func(sim, state, parameters)
            return state, parameters


After instantiating an Algorithm object, the user can populate its recipes by assigning tasks to each recipe. For example, the mean-field algorithm can be defined from an empty Algorithm object as:

.. code-block:: python

    from qclab import Algorithm
    import qclab.tasks as tasks
    from functools import partial

    # Create an empty Algorithm object.
    algorithm = Algorithm()
    # Populate the initialization recipe.
    algorithm.initialization_recipe = [
            tasks.initialize_variable_objects,
            tasks.initialize_norm_factor,
            tasks.initialize_z,
            tasks.update_h_q_tot,
    ]
    # Populate the update recipe.
    algorithm.update_recipe = [
        # Begin RK4 integration steps.
        # RK4 steps excluded for brevity.
        # End RK4 integration steps.
        tasks.update_wf_db_rk4,
        tasks.update_h_q
    ]
    # Populate the collect recipe.
    algorithm.collect_recipe = [
        tasks.update_t,
        tasks.update_dm_db_mf,
        tasks.update_quantum_energy,
        tasks.update_classical_energy,
        tasks.collect_t,
        tasks.collect_dm_db,
        tasks.collect_classical_energy,
        tasks.collect_quantum_energy,
    ]


Each recipe is executed by the method ``algorithm.execute_recipe``. The initialization recipe is executed once at the beginning of the simulation, the update recipe is executed at each time step of the simulation, and the collect recipe is executed once at the end of the simulation to gather and process results.


Mean Field Example
-------------------------------

As an example of a complete algorithm we include the source code for the mean-field algorithm below. This algorithm is defined in the ``qclab.algorithms.MeanField`` module and uses tasks from the ``qclab.tasks`` module to populate its recipes.

.. list-table:: FSSH Collected Observables
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Description
   * - ``quantum_energy``
     - The quantum energy of the system.
   * - ``classical_energy``
     - The classical energy of the system.
   * - ``dm_db``
     - The diabatic density matrix of the quantum subsystem.
   * - ``t``
     - The time points of the simulation.

.. dropdown:: View full source
   :icon: code

   .. literalinclude:: ../../src/qclab/algorithms/mean_field.py
      :language: python
      :linenos:
      :lines: 1-76

.. _fssh_source:

Surface Hopping Example
-------------------------------

As an additional example of a complete algorithm we include the source code for the fewest-switches surface hopping algorithm below. This algorithm is defined in the ``qclab.algorithms.FewestSwitchesSurfaceHopping`` module and uses tasks from the ``qclab.tasks`` module to populate its recipes.

.. list-table:: FSSH Collected Observables
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Description
   * - ``quantum_energy``
     - The quantum energy of the system.
   * - ``classical_energy``
     - The classical energy of the system.
   * - ``dm_db``
     - The diabatic density matrix of the quantum subsystem.
   * - ``t``
     - The time points of the simulation.


.. dropdown:: View full source
   :icon: code

   .. literalinclude:: ../../src/qclab/algorithms/fewest_switches_surface_hopping.py
      :language: python
      :linenos:
      :lines: 1-131

.. _ab_initio_fssh_source:

Ab Initio Surface Hopping Example
---------------------------------
As an example of an algorithm customized to Model objects defined in an adiabatic basis for compatibility with ab initio calculations, here we include the source code for the ab initio 
fewest-switches surface hopping algorithm implemented in the module ``qclab.algorithms.FewestSwitchesSurfaceHopping``. 

.. list-table:: Ab initio FSSH Collected Observables
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Description
   * - ``quantum_energy``
     - The quantum energy of the system.
   * - ``classical_energy``
     - The classical energy of the system.
   * - ``dm_adb``
     - The adiabatic density matrix of the quantum subsystem.
   * - ``t``
     - The time points of the simulation.


.. dropdown:: View full source
   :icon: code

   .. literalinclude:: ../../src/qclab/algorithms/fewest_switches_surface_hopping.py
      :language: python
      :linenos:
      :lines: 132-500