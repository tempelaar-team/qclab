.. _constants:

==========================
The Constants Object
==========================

A Constants object is an attribute bag with one extra feature: when an
attribute is set on it, an optional update function is called. QC Lab
uses a Constants object in three places — on the Simulation object, on
the Model object, and on the Algorithm object — to hold settings,
physical constants, and algorithm-specific options respectively. This
section describes the Constants object once and catalogs the three
usages so the differences are easy to keep straight.

The class itself lives at :class:`qclab.constants.Constants`.

----

What a Constants object does
============================

A Constants object stores user-set attributes by name and triggers a
single registered update function when any non-internal attribute is
written. The relevant excerpt from
:class:`qclab.constants.Constants` is:

.. code-block:: python

    class Constants:
        def __init__(self, update_function=None):
            self._updating = False
            self._init_complete = False
            self._update_function = update_function

        def __setattr__(self, name, value):
            super().__setattr__(name, value)
            if (not self._updating
                    and name not in {"_updating", "_update_function", "_init_complete"}
                    and self._init_complete
                    and self._update_function is not None):
                self._updating = True
                self._update_function()
                self._updating = False

        def get(self, name, default=None):
            return getattr(self, name, default)

Three properties of this design matter for users:

- The update function fires **only after** ``_init_complete`` is set
  to ``True``. During the constructor of the enclosing object (Model
  or Simulation), attributes are assigned without re-running the
  update function on every assignment.
- The update function is reentrancy-guarded by ``_updating``, so a
  ``__setattr__`` triggered from inside the update function does not
  recurse.
- :meth:`get` returns the attribute or a default; it is the
  recommended accessor when a constant may be absent from the
  object.

----

The three Constants objects
===========================

QC Lab uses three Constants objects with three different update
functions and three different roles:

.. list-table::
   :header-rows: 1
   :widths: 22 24 27 27

   * - Attribute
     - Owner
     - Update function
     - Holds
   * - ``sim.settings``
     - Simulation
     - none
     - Driver-level settings: ``tmax``, ``dt_update``, ``dt_collect``,
       ``num_trajs``, ``batch_size``, ``progress_bar``, ``debug``.
       Per-batch values such as ``tmax_n`` and ``t_collect`` are
       added by the dynamics core.
   * - ``model.constants``
     - Model
     - ``model.initialize_constants``
     - Physical constants of the Model object and any derived values
       that the Model's ``_init_*`` ingredients populate (e.g.,
       ``harmonic_frequency``, ``diagonal_linear_coupling``,
       ``num_quantum_states``, ``num_classical_coordinates``).
   * - ``algorithm.settings``
     - Algorithm
     - none
     - Algorithm-specific settings (e.g., ``gauge_fixing``,
       ``fssh_deterministic``, ``use_gauge_field_force``).

The differences between the three are mechanical, not conceptual: each
is a Constants object; the choice of update function is what makes the
Model object's Constants object re-initialize after every change while
the Simulation's and the Algorithm's stay quiet.

----

``sim.settings``: the Simulation object's Constants object
==========================================================

``sim.settings`` is constructed without an update function and is
populated from the Simulation object's default settings merged with
any user-provided overrides:

.. code-block:: python

    sim = Simulation()
    sim.settings.tmax = 5.0      # no re-initialization triggered

After the Simulation object is constructed, additional per-batch
attributes are written by the dynamics core
(:meth:`Simulation.initialize_timesteps <qclab.Simulation.initialize_timesteps>`)
— ``tmax_n``, ``dt_collect_n``, ``t_update``, ``t_collect``, and so
on. These are derived from ``tmax``, ``dt_update``, and ``dt_collect``
and are not meant to be edited directly.

The defaults populated by :class:`qclab.Simulation` are listed in
:ref:`Simulations <simulation>`.

----

``model.constants``: the Model object's Constants object
========================================================

The Model object's Constants object is the only one with an update
function. The update function is the Model's own
:meth:`initialize_constants <qclab.Model.initialize_constants>`,
which walks the Model's Ingredients list back-to-front and runs every
Ingredient whose name starts with ``_init_``:

.. code-block:: python

    def initialize_constants(self):
        for ingredient in self.ingredients[::-1]:
            if ingredient[0].startswith("_init_") and ingredient[1] is not None:
                ingredient[1](self, None)

This means that when a user changes a Model constant **after the
Model has been constructed**, every ``_init_*`` Ingredient runs again
and recomputes any derived values that depend on the changed
constant. For example:

.. code-block:: python

    from qclab.models import SpinBoson

    model = SpinBoson()
    # Changing l_reorg triggers initialize_constants(), which re-runs
    # _init_diagonal_linear_coupling and writes a fresh
    # diagonal_linear_coupling array onto model.constants.
    model.constants.l_reorg = 0.05

The same mechanism is what populates the standard constants
``num_quantum_states``, ``num_classical_coordinates``,
``classical_coordinate_mass``, and ``classical_coordinate_weight`` on
the Model object's Constants object.

Two cautions follow:

- Writing many constants in succession re-runs every ``_init_*``
  Ingredient on every write. When initializing a Model object from a
  large dictionary, pass the dictionary to the constructor
  (``SpinBoson(constants={...})``) so the re-initialization happens
  once at the end of the constructor, rather than on each
  attribute assignment.
- The flag ``_init_complete`` is set inside :meth:`Model.__init__`,
  so any constants set in the constructor are written before the
  update function would otherwise fire.

----

``algorithm.settings``: the Algorithm object's Constants object
===============================================================

The Algorithm object's Constants object is constructed without an
update function (the Algorithm class itself does not subclass
:class:`qclab.Model`). It holds settings specific to the Algorithm,
which the Algorithm's Tasks read directly. The default settings
exposed by the built-in Algorithms are:

.. list-table::
   :header-rows: 1
   :widths: 36 24 40

   * - Setting
     - Default
     - Algorithm
   * - ``fssh_deterministic``
     - ``False``
     - FSSH, FSSH (*ab initio*)
   * - ``gauge_fixing``
     - ``"sign_overlap"``
     - FSSH
   * - ``use_gauge_field_force``
     - ``False``
     - FSSH, FSSH (*ab initio*)
   * - ``update_wf_adb_eig_num_substeps``
     - ``10``
     - mean-field (*ab initio*), FSSH (*ab initio*)
   * - ``use_wf_overlaps_for_adb_connection``
     - varies
     - mean-field (*ab initio*), FSSH (*ab initio*)

Changing a value on ``algorithm.settings`` after the Algorithm has
been constructed does not trigger any callback — the change is simply
observed by whichever Tasks consult that setting at the next
invocation.

----

Reading a constant safely
=========================

When a Task or Ingredient consults a Constants object, the
recommended accessor is :meth:`get`:

.. code-block:: python

    fssh_deterministic = sim.algorithm.settings.get(
        "fssh_deterministic", False
    )

This returns the default when the constant is absent rather than
raising :class:`AttributeError`. The :meth:`get` method behaves like
:meth:`dict.get` and is preferable to a bare attribute access when
the calling code might run against a Model object or Algorithm
object configured without the constant in question.

----

Auto-generated reference
========================

.. autoclass:: qclab.constants.Constants
   :members:
   :undoc-members:
