.. _developing-algorithms:

==========================================
Developing Custom Algorithms
==========================================

This section describes how to add a new dynamics method to QC Lab by
subclassing :class:`qclab.Algorithm`. It is the Algorithm-side
companion to :ref:`Developing Models and Ingredients
<developing-models>`, and assumes that the
:ref:`Tasks <task>` and :ref:`Recipes <recipe>` sections have already
been read.

The procedure is the same as for the four built-in Algorithms: define
the three Recipes, declare any algorithm-specific settings, and write
new Tasks for the operations that do not yet exist in
:mod:`qclab.tasks`.

----

When subclassing an existing Algorithm is enough
================================================

Before writing a new Algorithm class, it is worth checking whether
the desired change can be expressed as a small edit to an existing
Algorithm. Two situations come up frequently:

- **Different settings.** If the change is only to one of the values
  on ``algorithm.settings`` (e.g., a different ``gauge_fixing``
  scheme for FSSH), instantiate the existing Algorithm with the
  new value and skip subclassing entirely:

  .. code-block:: python

      from qclab.algorithms import FewestSwitchesSurfaceHopping

      algorithm = FewestSwitchesSurfaceHopping(
          settings={"gauge_fixing": "phase_der_couple"}
      )

- **Extra side-effects on the existing Recipes.** If the change is
  to record an extra quantity or to insert a debug Task after every
  RK4 step, append the relevant Tasks to the existing Recipes
  in user code (see :ref:`Recipes <recipe>`). No new Algorithm class
  is needed.

A new Algorithm class is appropriate when the dynamics itself is
different — a new integrator, a new surface-hopping criterion, a new
decoherence correction — and not when the existing Tasks suffice but
need to be reordered or reconfigured.

----

The shape of a new Algorithm class
==================================

A new Algorithm class subclasses :class:`qclab.Algorithm` and
provides:

- A ``default_settings`` dictionary on the instance (set inside
  ``__init__``), which is merged with any user-supplied settings.
- Three class-level Recipe attributes —
  ``initialization_recipe``, ``update_recipe``, and
  ``collect_recipe`` — each a list of Tasks (or
  :func:`functools.partial` wrappings of Tasks).
- A docstring identifying the method and the basis it lives in.

The skeleton is:

.. code-block:: python

    from functools import partial
    from qclab.algorithm import Algorithm
    from qclab import tasks


    class MyAlgorithm(Algorithm):
        """Short description of the method and the basis it uses."""

        def __init__(self, settings=None):
            if settings is None:
                settings = {}
            self.default_settings = {
                # Algorithm-specific defaults go here.
            }
            super().__init__(self.default_settings, settings)

        initialization_recipe = [
            # Tasks executed once at t = 0.
        ]

        update_recipe = [
            # Tasks executed on every update time step.
        ]

        collect_recipe = [
            # Tasks executed on every collect time step.
        ]

The ``super().__init__`` call wires up the Constants object behind
``algorithm.settings`` and deep-copies the three Recipes onto the
instance (see :ref:`Recipes <recipe>` and
:ref:`The Constants Object <constants>` for the details).

----

Worked example: a fixed-step mean-field variant
================================================

The built-in mean-field algorithm uses RK4 to integrate the classical
coordinate. A pedagogical alternative is to use a single-step Euler
update; the example below packages that into a new Algorithm class
to illustrate the moving parts.

The Euler step is a new Task, because it is not in
:mod:`qclab.tasks`. The Task computes
:math:`z(t + \mathrm{d}t) = z(t) + \mathrm{d}t \, \dot{z}(t)` from
the existing ``classical_force`` and ``quantum_classical_force``
State entries:

.. code-block:: python

    def update_z_euler(sim, state, parameters,
                       *, z_name="z"):
        """Single Euler step for the complex-classical coordinate.

        Uses the classical force and the quantum-classical force
        already on the State object; writes back into z_name.
        """
        dt = sim.settings.dt_update
        z = state[z_name]
        # The forces are stored as derivatives w.r.t. zc, so the
        # time derivative is -i * (classical_force + qc_force) per the
        # complex-classical coordinate equations of motion.
        zdot = -1j * (state["classical_force"]
                      + state["quantum_classical_force"])
        state[z_name] = z + dt * zdot
        return state, parameters

The new Algorithm reuses every existing initialization and collect
Task; only the update Recipe is shortened to a single Euler step
followed by the standard wavefunction propagation:

.. code-block:: python

    from functools import partial
    from qclab.algorithm import Algorithm
    from qclab import tasks


    class MeanFieldEuler(Algorithm):
        """Mean-field dynamics with a single-step Euler integrator.

        Intended for pedagogy and short-time benchmarks; the time-step
        stability is much worse than the RK4 default.
        """

        def __init__(self, settings=None):
            if settings is None:
                settings = {}
            self.default_settings = {}
            super().__init__(self.default_settings, settings)

        initialization_recipe = [
            tasks.initialize_variable_objects,
            tasks.initialize_norm_factor,
            tasks.initialize_z,
            tasks.update_h_q_tot,
        ]

        update_recipe = [
            tasks.update_classical_force,
            tasks.update_quantum_classical_force,
            update_z_euler,
            tasks.update_wf_db_rk4,
            tasks.update_h_q_tot,
        ]

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

The class can then be used in place of the built-in mean-field
algorithm:

.. code-block:: python

    from qclab import Simulation
    from qclab.models import SpinBoson
    from qclab.dynamics import serial_driver

    sim = Simulation(settings={"dt_update": 0.0001})  # Euler needs a smaller step
    sim.model = SpinBoson()
    sim.algorithm = MeanFieldEuler()
    sim.initial_state["wf_db"] = np.array([1, 0], dtype=complex)
    data = serial_driver(sim)

----

Writing new Tasks for a new Algorithm
=====================================

A new Algorithm typically introduces one or more new Tasks. The
checklist for a new Task is:

- The signature is ``f(sim, state, parameters, **kwargs)`` (or
  ``f(sim, state, parameters)`` when there are no keyword
  arguments).
- The return value is ``(state, parameters)``.
- Every State or Parameters key the Task reads or writes is exposed
  as a ``*_name`` keyword argument with a default that matches the
  standard names listed in :ref:`Conventions <conventions>`.
- The docstring uses the Napoleon sections recognized by the QC Lab
  Sphinx configuration: ``Optional Keyword Arguments``, ``Reads``,
  ``Writes``, ``Constants and Settings``, ``Ingredients``,
  ``Shapes and dtypes``. See
  :ref:`State and Parameters Objects <state-and-parameters>` for
  the full template.
- The Task is vectorized along the batch axis. The leading dimension
  of every State entry is ``sim.settings.batch_size``; the Task
  should be written to operate on the whole batch at once with NumPy
  broadcasting.

Initialization Tasks create State entries; update Tasks transform
them in place; collect Tasks copy a chosen State entry into
``state["output_dict"]``. The naming convention is
``initialize_*``, ``update_*``, and ``collect_*`` for the three
categories.

----

Algorithm-specific settings
===========================

If the new Algorithm needs settings of its own — a threshold for a
new hopping criterion, a flag for an alternative decoherence
correction — the values go into ``self.default_settings`` and are
read from ``sim.algorithm.settings`` by the Tasks that consult them.

.. code-block:: python

    class MyAlgorithm(Algorithm):
        def __init__(self, settings=None):
            if settings is None:
                settings = {}
            self.default_settings = {
                "decoherence_threshold": 1e-3,
                "use_branching_correction": True,
            }
            super().__init__(self.default_settings, settings)

A Task that reads one of these settings does so via the Constants
object's :meth:`get` method (see :ref:`The Constants Object
<constants>`):

.. code-block:: python

    threshold = sim.algorithm.settings.get("decoherence_threshold", 1e-3)

Using :meth:`get` with a default makes the Task robust to running
inside an Algorithm that has not declared the setting.

----

Testing a new Algorithm
=======================

Two cheap checks for a new Algorithm:

#. **Conservation.** Run the new Algorithm on a Model object whose
   built-in Algorithm conserves a quantity (energy in the
   closed-system Models, the wavefunction norm in every Model) and
   verify the same conservation holds.
#. **Limit reproduction.** When the new Algorithm has a parameter
   that reduces it to a known existing Algorithm, run both at the
   reducing parameter value and check that the trajectory-averaged
   observables agree.

For deterministic comparisons set ``fssh_deterministic = True``
(or its equivalent for the new Algorithm) and pass an explicit
``seeds`` array to the Driver; see :ref:`Parallelization and
Reproducibility <parallelization>`.
