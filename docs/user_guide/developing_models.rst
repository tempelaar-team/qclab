.. _developing-models:

==========================================
Developing Models and Ingredients
==========================================

This page is a guided tour for users who want to add new physics to
QC Lab — either by tweaking an existing model or by writing a new one
from scratch. Read it after :ref:`Ingredients <ingredient>` and
:ref:`Models <model>`, both of which describe what the building blocks
look like in isolation.

Decision tree: subclass an existing model, or write a new one?
==============================================================

Before writing a new ingredient, check whether the change you want can
be expressed as a different set of *constants* on top of the existing
parametric ingredients. The stock ingredients in :mod:`qclab.ingredients`
are intentionally generic:

- ``h_qc_diagonal_linear`` works for *any* ``diagonal_linear_coupling``
  matrix.
- ``h_c_harmonic`` works for *any* ``harmonic_frequency`` array.
- ``h_q_two_level`` works for *any* two-level Hamiltonian set by
  ``two_level_00``, ``two_level_11``, ``two_level_01_re`` /
  ``two_level_01_im``.

If the *functional form* of the physics stays the same and only the
*distribution of parameters* changes, you do not need a new ingredient.
Subclass the existing model and override the relevant ``_init_*``
methods.

If the functional form genuinely changes — for instance, off-diagonal
coupling where only diagonal coupling exists, or a position-dependent
coupling where only linear coupling exists — write a new ingredient.

A useful test: if your new physics can be described by the existing
ingredient's docstring, modify constants. If it cannot, write a new
ingredient.

Example: subclassing for a different spectral density
-----------------------------------------------------

A spin-boson model with a Debye spectral density still uses
``h_qc_diagonal_linear`` and ``h_c_harmonic``. Only the way the
``harmonic_frequency`` and ``diagonal_linear_coupling`` arrays are
populated changes. A subclass is sufficient:

.. code-block:: python

    import numpy as np
    from qclab.models import SpinBoson

    class SpinBosonDebye(SpinBoson):
        """Spin-boson with a Debye spectral density.

        J(w) = 2 * l_reorg * w_D * w / (w**2 + w_D**2)
        """

        def __init__(self, constants=None):
            if constants is None:
                constants = {}
            base_defaults = SpinBoson({}).default_constants
            self.default_constants = {**base_defaults, "w_D": 0.5}
            super().__init__({**self.default_constants, **constants})

        def _init_h_c(self, parameters, **kwargs):
            A = self.constants.A
            w_D = self.constants.w_D
            w_max = 10.0 * w_D
            self.constants.harmonic_frequency = np.linspace(
                w_max / A, w_max, A
            )

        def _init_h_qc(self, parameters, **kwargs):
            A = self.constants.A
            w_D = self.constants.w_D
            l_reorg = self.constants.l_reorg
            boson_mass = self.constants.boson_mass
            h = self.constants.classical_coordinate_weight
            w = self.constants.harmonic_frequency
            dw = w[1] - w[0]
            J_w = 2.0 * l_reorg * w_D * w / (w ** 2 + w_D ** 2)
            g = np.sqrt(2.0 * J_w * dw / np.pi)
            self.constants.diagonal_linear_coupling = np.zeros((2, A))
            self.constants.diagonal_linear_coupling[0] = (
                g / np.sqrt(2.0 * boson_mass * h)
            )
            self.constants.diagonal_linear_coupling[1] = -(
                g / np.sqrt(2.0 * boson_mass * h)
            )

The ingredient list is inherited from ``SpinBoson``; the subclass only
swaps in the new initializer logic.

----

The new-model checklist
=======================

When the functional form of the physics is genuinely new, write a new
``Model`` subclass. The minimum viable model satisfies the following
checklist:

#. **Subclass** :class:`qclab.Model`. Place the file in
   ``src/qclab/models/`` if you intend to upstream it; a self-contained
   file outside the package is fine for prototyping.
#. **Set ``default_constants``** in ``__init__`` and forward them via
   ``super().__init__(self.default_constants, constants)``.
#. **Set the performance flags** ``self.update_h_q`` and
   ``self.update_dh_qc_dzc`` to ``False`` if those quantities do not
   depend on ``z`` (a significant speedup) and to ``True`` otherwise.
#. **Provide an ``_init_model`` initializer** that derives the mandatory
   constants ``num_quantum_states``, ``num_classical_coordinates``,
   ``classical_coordinate_mass``, and ``classical_coordinate_weight``.
   Optional ``_init_h_q`` / ``_init_h_qc`` / ``_init_h_c`` initializers
   can be added to derive ingredient-specific constants.
#. **Define an ``ingredients`` class attribute** that lists the
   ingredients the model uses, including the ``_init_*`` initializers.
   Reuse stock ingredients from :mod:`qclab.ingredients` where possible.
#. **Vectorize new ingredients** over the batch axis. Either hand-code
   the ``(B, ...)`` shape or use the
   :func:`~qclab.functions.vectorize_ingredient` decorator from
   :mod:`qclab.functions`.
#. **Make sparse gradients return ``(inds, mels, shape)``** in that
   order; ``inds`` should come from ``np.where`` on a dense array. See
   :ref:`Sparse Quantum-Classical Gradients <ingredient>`.
#. **Cite a reference** in the class docstring, when relevant.

Use :class:`~qclab.models.SpinBoson` and
:class:`~qclab.models.HolsteinLattice` as canonical examples.

----

Worked example: a linear vibronic coupling (LVC) model
======================================================

The example below assembles a two-state, two-mode linear-vibronic-coupling
model with a conical intersection. The point of the example is to walk
through every step of the new-model checklist. It does not use
``init_classical_wigner_coherent_state`` or any other off-the-shelf
sampler that requires extra constants — the focus is on the model and
its ingredients, not on initial conditions.

.. code-block:: python

    import numpy as np
    from qclab import Simulation
    from qclab.model import Model
    from qclab import ingredients
    from qclab import functions
    from qclab.algorithms import MeanField
    from qclab.dynamics import serial_driver

    # ------------------- novel ingredients ------------------------------

    @functions.vectorize_ingredient
    def h_qc_lvc(model, parameters, **kwargs):
        """Linear vibronic-coupling quantum-classical Hamiltonian.

        H_qc = kappa_1 q_1 |1><1| + kappa_2 q_1 |2><2|
                  + lambda q_2 (|1><2| + |2><1|)

        Keyword Args
        ------------
        z : ndarray, shape (C,), complex128

        Model Constants
        ---------------
        kappa : (2,) array of tuning-mode slopes.
        lam   : float, off-diagonal coupling strength.

        Returns
        -------
        h_qc : ndarray, shape (N, N), complex128
        """
        z = kwargs["z"]
        kappa = model.constants.kappa
        lam = model.constants.lam
        m = model.constants.classical_coordinate_mass
        h = model.constants.classical_coordinate_weight
        q = functions.z_to_q(z, m, h)
        q_tune, q_couple = q[0], q[1]
        h_qc = np.zeros((2, 2), dtype=complex)
        h_qc[0, 0] = kappa[0] * q_tune
        h_qc[1, 1] = kappa[1] * q_tune
        h_qc[0, 1] = lam * q_couple
        h_qc[1, 0] = np.conj(h_qc[0, 1])
        return h_qc

    @functions.make_ingredient_sparse
    @functions.vectorize_ingredient
    def dh_qc_dzc_lvc(model, parameters, **kwargs):
        """Sparse z*-derivative of h_qc_lvc."""
        kappa = model.constants.kappa
        lam = model.constants.lam
        m = model.constants.classical_coordinate_mass
        h = model.constants.classical_coordinate_weight
        # dq/dz* = sqrt(1 / (2 m h))
        dq_dzc = 1.0 / np.sqrt(2.0 * m * h)
        out = np.zeros((2, 2, 2), dtype=complex)
        out[0, 0, 0] = kappa[0] * dq_dzc[0]
        out[0, 1, 1] = kappa[1] * dq_dzc[0]
        out[1, 0, 1] = lam * dq_dzc[1]
        out[1, 1, 0] = np.conj(out[1, 0, 1])
        return out

    # ------------------- the Model class --------------------------------

    class LinearVibronicCoupling(Model):
        """Two-state, two-mode linear vibronic coupling model.

        Mode 0 ("tuning") couples diagonally to the two electronic states
        with slopes ``kappa[0]`` and ``kappa[1]``. Mode 1 ("coupling")
        couples off-diagonally with strength ``lam``. Both modes are
        harmonic oscillators with frequency ``w``.

        Reference: Koppel, Domcke & Cederbaum, *Adv. Chem. Phys.* **57**,
        59 (1984).
        """

        def __init__(self, constants=None):
            if constants is None:
                constants = {}
            self.default_constants = {
                "kBT":   1.0,
                "w":     1.0,
                "kappa": np.array([0.5, -0.5]),
                "lam":   0.3,
                "mass":  1.0,
            }
            super().__init__(self.default_constants, constants)
            self.update_h_q = False
            self.update_dh_qc_dzc = False

        def _init_model(self, parameters, **kwargs):
            self.constants.num_quantum_states = 2
            self.constants.num_classical_coordinates = 2
            self.constants.classical_coordinate_mass = (
                self.constants.mass * np.ones(2)
            )
            self.constants.harmonic_frequency = (
                self.constants.w * np.ones(2)
            )
            self.constants.classical_coordinate_weight = (
                self.constants.harmonic_frequency.copy()
            )

        def _init_h_q(self, parameters, **kwargs):
            self.constants.two_level_00 = 0.0
            self.constants.two_level_11 = 0.0
            self.constants.two_level_01_re = 0.0
            self.constants.two_level_01_im = 0.0

        ingredients = [
            ("h_q",            ingredients.h_q_two_level),
            ("h_qc",           h_qc_lvc),
            ("h_c",            ingredients.h_c_harmonic),
            ("dh_qc_dzc",      dh_qc_dzc_lvc),
            ("dh_c_dzc",       ingredients.dh_c_dzc_harmonic),
            ("init_classical", ingredients.init_classical_wigner_harmonic),
            ("hop",            ingredients.hop_harmonic),
            ("_init_h_q",      _init_h_q),
            ("_init_model",    _init_model),
        ]

    # ------------------- run --------------------------------------------

    sim = Simulation({
        "tmax": 30.0, "dt_update": 0.005, "dt_collect": 0.1,
        "num_trajs": 200, "batch_size": 100, "progress_bar": False,
    })
    sim.model = LinearVibronicCoupling()
    sim.algorithm = MeanField()
    sim.initial_state["wf_db"] = np.array([1.0 + 0j, 0.0 + 0j])
    data = serial_driver(sim)

The sparse gradient is implemented analytically and decorated with
``@make_ingredient_sparse``, so QC Lab does not fall back to
finite-difference gradients. Note that ``update_h_q`` and
``update_dh_qc_dzc`` are both ``False`` because neither depends on
``z`` after the constants are derived.

----

Things to double-check
======================

When debugging a new model, the following are the most common sources of
incorrect results:

- Forgetting to set ``classical_coordinate_weight`` — the complex
  coordinate ``z`` is not well-defined without it.
- Returning a dense gradient from ``dh_qc_dzc`` instead of the sparse
  ``(inds, mels, shape)`` triple. Wrap with
  :func:`~qclab.functions.make_ingredient_sparse` to convert
  automatically, or build the indices manually with ``np.where``.
- Setting ``update_dh_qc_dzc = False`` on a model whose gradient does
  depend on a constant that you intend to vary at runtime; the cached
  gradient becomes stale.
- Hard-coding the batch size from ``len(z)`` rather than reading
  ``sim.settings.batch_size``. ``len(z)`` is correct in ingredients
  (which see only ``model``, ``parameters``, ``kwargs``), but tasks
  must use ``sim.settings.batch_size``.

When in doubt, look at how :class:`~qclab.models.SpinBoson` and
:class:`~qclab.models.TullyProblemOne` are written. They cover the two
common cases: a many-mode harmonic bath and a single-coordinate scattering
problem.
