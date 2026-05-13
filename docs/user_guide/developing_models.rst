.. _developing-models:

==========================================
Developing Models and Ingredients
==========================================

This section describes how to add new physics to QC Lab by either
adapting an existing Model object or writing a new one. It is intended
to be read after the :ref:`Ingredients <ingredient>` and :ref:`Models
<model>` sections, which describe what the building blocks look like
in isolation.

When to subclass an existing Model object
=========================================

Before writing a new ingredient, it is worth checking whether the
desired change can be expressed by changing a Model object's constants
on top of the existing ingredients. Several of the built-in ingredients
in :mod:`qclab.ingredients` are parametric and can accommodate a range
of physical settings without modification.

For example:

- ``h_qc_diagonal_linear`` works for any ``diagonal_linear_coupling``
  matrix.
- ``h_c_harmonic`` works for any ``harmonic_frequency`` array.
- ``h_q_two_level`` works for any two-level Hamiltonian set by
  ``two_level_00``, ``two_level_11``, ``two_level_01_re``, and
  ``two_level_01_im``.

When the functional form of the physics is the same as that of an
existing ingredient and only the distribution of parameters changes,
subclassing an existing Model object and overriding the relevant
``_init_*`` methods is often sufficient. When the functional form is
different — for example, off-diagonal coupling where only diagonal
coupling exists, or a position-dependent coupling where only a linear
coupling exists — a new ingredient is required.

Example: subclassing for a different spectral density
-----------------------------------------------------

A spin-boson model with a Debye spectral density still uses
``h_qc_diagonal_linear`` and ``h_c_harmonic``; only the way the
``harmonic_frequency`` and ``diagonal_linear_coupling`` arrays are
populated changes. A subclass of :class:`~qclab.models.SpinBoson` that
overrides the corresponding initializers is sufficient:

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

The ingredient list is inherited from
:class:`~qclab.models.SpinBoson`; only the initializer logic is changed.

----

A new Model object from scratch
===============================

When the functional form of the physics differs from any of the
ingredients shipped with QC Lab, a new Model object is needed. The
items below summarize what such a Model object needs to provide.

#. **Subclass** :class:`qclab.Model`. The file can live in
   ``src/qclab/models/`` if it is intended to be upstreamed; a
   self-contained file outside the package is fine otherwise.
#. **Set** ``default_constants`` in ``__init__`` and forward them via
   ``super().__init__(self.default_constants, constants)``.
#. **Set the performance flags** ``self.update_h_q`` and
   ``self.update_dh_qc_dzc`` to ``False`` if those quantities do not
   depend on the classical coordinate ``z`` and to ``True`` otherwise.
   Setting either flag to ``False`` when the underlying quantity does
   depend on ``z`` produces a stale cache and incorrect results.
#. **Provide an** ``_init_model`` **initializer** that derives the
   mandatory constants ``num_quantum_states``,
   ``num_classical_coordinates``, ``classical_coordinate_mass``, and
   ``classical_coordinate_weight``. Optional ``_init_h_q``,
   ``_init_h_qc``, ``_init_h_c`` initializers can be added to derive
   ingredient-specific constants.
#. **Define an** ``ingredients`` **class attribute** that lists the
   ingredients the model uses, including the ``_init_*`` initializers.
   Reuse ingredients from :mod:`qclab.ingredients` whenever an existing
   ingredient covers the desired physics.
#. **Vectorize new ingredients over the batch axis.** Either hand-code
   the ``(B, ...)`` shape or use the
   :func:`~qclab.functions.vectorize_ingredient` decorator from
   :mod:`qclab.functions`.
#. **Return sparse gradients as** ``(inds, mels, shape)`` in that
   order. ``inds`` should come from ``np.where`` on a dense array. See
   the :ref:`Ingredients <ingredient>` section for details.
#. **Cite the reference** for the physics in the class docstring when
   one exists.

:class:`~qclab.models.SpinBoson` and
:class:`~qclab.models.HolsteinLattice` cover the case of a many-mode
harmonic bath; :class:`~qclab.models.TullyProblemOne` covers a
single-coordinate scattering problem. Both can be used as a starting
template.

----

Example: a linear vibronic coupling model
=========================================

The example below assembles a two-state, two-mode linear-vibronic-coupling
model with a conical intersection. It shows the workflow described
above and is intended as a worked example, not a reference
implementation.

.. code-block:: python

    import numpy as np
    from qclab import Simulation
    from qclab.model import Model
    from qclab import ingredients
    from qclab import functions
    from qclab.algorithms import MeanField
    from qclab.dynamics import serial_driver

    # ---- novel ingredients ----------------------------------------------

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

    # ---- the Model class ------------------------------------------------

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

    # ---- run ------------------------------------------------------------

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
finite-difference gradients. Both performance flags ``update_h_q`` and
``update_dh_qc_dzc`` are set to ``False`` because the corresponding
quantities do not depend on ``z`` once the constants are derived.

----

Common sources of incorrect results
===================================

When debugging a new Model object, the following are recurring sources
of incorrect results. The list is representative, not exhaustive.

- Forgetting to set ``classical_coordinate_weight``. The complex
  coordinate ``z`` is not well defined without it.
- Returning a dense gradient from ``dh_qc_dzc`` instead of the sparse
  ``(inds, mels, shape)`` triple. Wrap with
  :func:`~qclab.functions.make_ingredient_sparse` to convert
  automatically, or build the indices manually with ``np.where``.
- Setting ``update_dh_qc_dzc = False`` on a model whose gradient depends
  on a constant that is varied at runtime. The cached gradient becomes
  stale.
- Reading the batch size from ``len(z)`` rather than from
  ``sim.settings.batch_size`` inside a task. The latter is the
  canonical source of the batch size in a task body.
