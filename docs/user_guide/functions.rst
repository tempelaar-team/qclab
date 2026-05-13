.. _functions:

==========================
Low-level Functions
==========================

The :mod:`qclab.functions` module collects the low-level numerical
helpers used by the built-in ingredients, tasks, and algorithms. Users
writing new physics, in particular new ingredients or tasks, will find
the coordinate conversions, sparse-gradient inner product, and
gauge-fixing routines documented in this section.

For the conceptual basis behind the complex-classical-coordinate
formalism see the :ref:`Coordinates <coordinates>` section.

----

Coordinate conversions
======================

QC Lab integrates classical degrees of freedom in the complex
coordinate ``z``. The functions below convert between ``z`` and the
real-valued phase-space pair ``(q, p)``, including for gradients.

.. note::

    Using these helpers rather than re-deriving the relationship
    between ``z`` and ``(q, p)`` inline is recommended; hand-rolled
    conversions are a common source of incorrect factors involving
    :math:`\sqrt{2 m h}`.

.. automodule:: qclab.functions
   :members: z_to_q, z_to_p, qp_to_z, dqdp_to_dzc, dzdzc_to_dqdp
   :undoc-members:
   :member-order: bysource
   :no-value:

----

Linear-algebra helpers
======================

Wrappers around batched matrix-vector and basis-change operations.
``transform_vec`` and ``transform_mat`` are used by the built-in
algorithms to switch between the diabatic and adiabatic bases.

.. automodule:: qclab.functions
   :members: batch_matvec, transform_vec, transform_mat
   :undoc-members:
   :member-order: bysource
   :no-value:
   :no-index:

----

RK4 integration kernels
=======================

The fourth-order Runge–Kutta integration of the classical coordinate is
factored into two summation kernels, each decorated with ``@njit``.
These are called by the corresponding update tasks
:func:`update_z_rk4_k123 <qclab.tasks.update_tasks.update_z_rk4_k123>`
and :func:`update_z_rk4_k4 <qclab.tasks.update_tasks.update_z_rk4_k4>`.

.. automodule:: qclab.functions
   :members: update_z_rk4_k123_sum, update_z_rk4_k4_sum
   :undoc-members:
   :member-order: bysource
   :no-value:
   :no-index:

----

Decorators for ingredients
==========================

Two decorators are provided to support the implementation of new
ingredients.

.. automodule:: qclab.functions
   :members: vectorize_ingredient, make_ingredient_sparse
   :undoc-members:
   :member-order: bysource
   :no-value:
   :no-index:

``@vectorize_ingredient`` turns a single-trajectory ingredient into a
batch-aware ingredient by looping over the trajectory axis and
broadcasting. It does not provide a performance gain over a
hand-vectorized implementation. See the :ref:`Vectorization
<ingredient>` discussion in the Ingredients section for examples.

``@make_ingredient_sparse`` turns a dense-tensor ingredient (for
example, a gradient that returns a full ``(B, C, N, N)`` array) into
the sparse ``(inds, mels, shape)`` form expected by the algorithms.
The indices come from ``np.where`` on the dense tensor.

----

Sparse-gradient inner product
=============================

When an ingredient returns its gradient in the sparse ``(inds, mels,
shape)`` form, contracting it against a wavefunction (to compute, for
example, a quantum-classical force) is performed by
``calc_sparse_inner_product``. This function consumes the triple
returned by ``make_ingredient_sparse``.

.. automodule:: qclab.functions
   :members: calc_sparse_inner_product
   :undoc-members:
   :member-order: bysource
   :no-value:
   :no-index:

----

JIT kernels for the harmonic ingredients
========================================

Two of the built-in harmonic ingredients have ``@njit`` kernels for
their inner loops. They are exposed below for completeness; the
corresponding ingredients (``dh_c_dzc_harmonic``,
``h_qc_diagonal_linear``) are the entry points used by Model objects.

.. automodule:: qclab.functions
   :members: dh_c_dzc_harmonic_jit, h_qc_diagonal_linear_jit
   :undoc-members:
   :member-order: bysource
   :no-value:
   :no-index:

----

Sampling
========

Helpers for drawing initial classical coordinates and for resolving the
classical part of an FSSH hop numerically when no analytical hop
ingredient is provided.

.. automodule:: qclab.functions
   :members: gen_sample_gaussian, numerical_fssh_hop
   :undoc-members:
   :member-order: bysource
   :no-value:
   :no-index:

----

Gauge fixing and rescaling
==========================

Adiabatic-basis algorithms need to keep the eigenvectors of
``h_q_tot`` continuous as the classical coordinate evolves, and FSSH
needs a direction in which to rescale the classical momentum after a
hop. Both pieces of machinery live here.

.. automodule:: qclab.functions
   :members: analytic_der_couple_phase, calc_resc_dir_z_fssh
   :undoc-members:
   :member-order: bysource
   :no-value:
   :no-index:
