.. _ingredient:

==========================
Ingredients
==========================

Ingredients are functions that encode the physics of a model. QC Lab is designed to be operational with a minimal set of ingredients that describe the Hamiltonian of the system which is assumed to have the form:

.. math::

    H(q,p) = \hat{H}_{\mathrm{q}} + \hat{H}_{\mathrm{q-c}}(q) + H_{\mathrm{c}}(q,p)

where :math:`\hat{H}_\mathrm{q}` is the quantum Hamiltonian, :math:`\hat{H}_{\mathrm{q-c}}(q)` is the quantum-classical coupling Hamiltonian, and :math:`H_{\mathrm{c}}(q,p)` is the classical Hamiltonian. 
As discussed in the :ref:`Model <model>` section, models defined in an adiabatic basis also provide the derivative coupling tensor :math:`D_{nij}(q)=\langle i(q)\vert\partial_{n}\vert j(q)\rangle`.

A generic ingredient has the form:

.. code-block:: python

    def ingredient_specifier(model, parameters, **kwargs):
        # Get any keyword arguments, with default values if not provided.
        kwarg1 = kwargs.get('kwarg1', default_value1)
        # Compute the ingredient using attributes of the model and parameters objects.
        constants = model.constants
        # Return the computed ingredient.
        return ingredient

where ``model`` is a Model object which contains all the constants of the model, ``parameters`` is a dictionary containing time-dependent parameters of the simulation, and ``**kwargs`` are any additional keyword arguments that are specific to that ingredient type. 

Ingredients in QC lab can come in different variations, for example the quantum Hamiltonian ingredient could describe a two-level system, a nearest-neighbor lattice, or a more complicated Hamiltonian. The type and variety of an ingredient is specified in its name which follows the convention ``<ingredient_type>_<variety>``. For example, the quantum Hamiltonian ingredient for a two-level system is named ``h_q_two_level`` where ``h_q`` indicates that it is a quantum Hamiltonian ingredient and ``two_level`` indicates that it describes a two-level system. Likewise the classical Hamiltonian ingredient for a harmonic oscillator is named ``h_c_harmonic`` where ``h_c`` indicates that it is a classical Hamiltonian ingredient and ``harmonic`` indicates that it describes a harmonic oscillator.



Ingredients can be included in a model by appending them to the model's ``ingredients`` attribute, which is a list of tuples where each tuple contains the name of the ingredient and the ingredient function itself. Because the ingredients list is read back to front, appending an ingredient is sufficient to overwrite any existing ingredient with the same name. For example, to include a custom quantum Hamiltonian ingredient in a model, one would do:

.. code-block:: python

    def h_q_custom(model, parameters, **kwargs):
        # Custom quantum Hamiltonian implementation.
        h_q = ...
        return h_q

    model.ingredients.append(("h_q", h_q_custom))


The minimal set of ingredients required to run a simulation are: 

- A quantum Hamiltonian ingredient, named ``h_q``.
- A classical Hamiltonian ingredient, named ``h_c``.
- A quantum-classical coupling Hamiltonian ingredient, named ``h_qc``.
- A derivative coupling tensor ingredient (only required for models defined in an adiabatic basis), named ``derivative_coupling_dzc``.

Additional optional ingredients that make the simulation more efficient or accurate are:

- An initialization function for the classical coordinates, named ``init_classical``.
- A gradient of the classical Hamiltonian with respect to the conjugate of the :ref:`complex classical coordinates <coordinates>`, named ``dh_c_dzc``.
- A gradient of the quantum-classical coupling Hamiltonian with respect to the conjugate of the :ref:`complex classical coordinates <coordinates>`, named ``dh_qc_dzc``.
- A hopping function for surface hopping algorithms, named ``hop``.
- An ab initio property calculator that calculates physical properties with an interface to an electronic structure theory code, named ``ab_initio_property_calculator``.


Vectorization
--------------------------

By default, QC Lab assumes that ingredients are implemented in a vectorized fashion. This means that, rather than constructing the respective term for an individual trajectory, each ingredient constructs the term for all trajectories in a batch at once. Additionally, any keyword argument associated with an ingredient that is a NumPy array will have an initial trajectory dimension. For example, the quantum-classical Hamiltonian ingredient ``h_qc`` has a keyword argument ``z`` which is the :ref:`complex classical coordinate <coordinates>`. If the batch size is 100, then ``z`` will have shape ``(100, model.constants.num_classical_coordinates)`` where ``model.constants.num_classical_coordinates`` is the number of classical coordinates in the model. The output of the ``h_qc`` ingredient will then have shape ``(100, model.constants.num_quantum_states, model.constants.num_quantum_states)`` where ``model.constants.num_quantum_states`` is the number of quantum states in the model.

Rather than implementing this vectorization yourself, you can use the ``@vectorize_ingredient`` decorator provided in the ``qclab.functions`` module. This decorator will automatically vectorize an ingredient that is implemented for a single trajectory (i.e., without any batch dimension). For example, the following implementation of the quantum Hamiltonian ingredient for a two-level system is not vectorized:

.. code-block:: python

    from qclab.functions import vectorize_ingredient

    @vectorize_ingredient
    def h_q_two_level(model, parameters, **kwargs):
        """
        Quantum Hamiltonian for a two-level system.
        """
        e_0 = model.constants.two_level_00
        e_1 = model.constants.two_level_11
        v = model.constants.two_level_01_re + 1j * model.constants.two_level_01_im
        h_q = np.zeros((2, 2), dtype=complex)
        h_q[0, 0] = e_0
        h_q[1, 1] = e_1
        h_q[0, 1] = v
        h_q[1, 0] = np.conj(v)
        return h_q

The ingredient can then be included in a model as:

.. code-block:: python

    model.ingredients.append(("h_q", h_q_two_level))


.. note::

    The vectorization decorator promotes compatibility and ease of use but does not offer any performance gain. To realize a performance gain, users
    should hard-code vectorization into Ingredients and leverage NumPy vectorization.


Alternatively, vectorization can be implemented manually. For example, the following implementation of the quantum Hamiltonian ingredient for a two-level system is manually vectorized as:


.. code-block:: python


    def h_q_two_level(model, parameters, **kwargs):
        """
        Quantum Hamiltonian for a two-level system.
        """
        batch_size = kwargs["batch_size"]
        e_0 = model.constants.two_level_00
        e_1 = model.constants.two_level_11
        v = model.constants.two_level_01_re + 1j * model.constants.two_level_01_im
        h_q = np.zeros((2, 2), dtype=complex)
        h_q[0, 0] = e_0
        h_q[1, 1] = e_1
        h_q[0, 1] = v
        h_q[1, 0] = np.conj(v)
        # Here we use np.broadcast_to to add the batch dimension to the output.
        # This is only possible because the output does not depend on any trajectory-specific
        # information.
        return np.broadcast_to(h_q, (batch_size, 2, 2))



Sparse Quantum-Classical Gradients
---------------------------------

If left unspecified, the gradient of the quantum-classical Hamiltonian will be calculated numerically using finite differences. This can be computationally expensive and may introduce inaccuracies. This can be avoided by providing an analytical implementation of the gradient ingredient ``dh_qc_dzc``. In QC Lab, we implement this gradient in a sparse manner, meaning that we only compute the non-zero elements of the gradient matrix. This reduces the potentially large gradient tensor with shape ``(sim.settings.batch_size, model.constants.num_classical_coordinates, model.constants.num_quantum_states, model.constants.num_quantum_states)`` to a list of non-zero elements, ``mels``, their indices ``inds``, and the shape of the full tensor ``shape``. 

In practice, the ordering of the matrix elements can have a dramatic impact on performance due to memory access patterns. Therefore, we recommend using ``numpy.where`` to determine the indices of the non-zero elements. For example, the following implementation of the gradient of the quantum-classical Hamiltonian for a spin-boson model computes only the non-zero elements of the gradient:

.. code-block:: python

    @vectorize_ingredient
    def dh_qc_dzc_spinboson(model, parameters, **kwargs):
        z = kwargs["z"]
        A = model.constants.get("A")
        diag_coupling = model.constants.diagonal_linear_coupling
        out = np.zeros((A, 2, 2), dtype=complex)
        for i in range(A):
            out[i, 0, 0] = diag_coupling[0, i]
            out[i, 1, 1] = diag_coupling[1, i]
        inds = np.where(out != 0)
        mels = out[inds]
        shape = (len(z), A, 2, 2)
        return inds, mels, shape



An ingredient that returns dense tensor output can be automatically converted to returning a sparse tensor by invoking the ``@make_ingredient_sparse`` 
decorator from the ``qclab.functions`` module. 
This decorator determines the non-zero elements of a dense tensor, their indices, and the shape of the full tensor. 
For example, the following implementation of the gradient of the quantum-classical Hamiltonian for a spin-boson model is dense, 
but made sparse by the decorator:

.. code-block:: python

    from qclab.functions import make_ingredient_sparse

    @make_ingredient_sparse
    @vectorize_ingredient
    def dh_qc_dzc_spinboson(model, parameters, **kwargs):
        z = kwargs["z"]
        A = model.constants.get("A")
        diag_coupling = model.constants.diagonal_linear_coupling
        out = np.zeros((A, 2, 2), dtype=complex)
        for i in range(A):
            out[i, 0, 0] = diag_coupling[0, i]
            out[i, 1, 1] = diag_coupling[1, i]
        return out

    
Of course, the most efficient implementation is one that is both analytical and sparse without invoking the decorator. This is what is implemented in the ingredient ``ingredients.dh_qc_dzc_diagonal_linear`` which is included in the Spin-Boson model by default.



Ingredients in QC Lab
---------------------------------

The built-in ingredients in QC Lab can be found in the ``qclab.ingredients`` module.

.. note::

   All ingredients assume that the Model object has a minimal set of constants including ``num_quantum_states`` 
   (the number of quantum states) and ``num_classical_coordinates`` (the number of classical coordinates), ``classical_coordinate_mass`` 
   (the mass of the classical coordinates), and ``classical_coordinate_weight`` (the weight of the classical coordinates). 
   These constants are discussed in :ref:`Models <model>`. For brevity we exclude explicit mention of these constants in the ingredient documentation.


.. automodule:: qclab.ingredients
   :members:
   :undoc-members:
   :member-order: bysource
   :imported-members:
   :exclude-members: __all__, __doc__, __annotations__, QCLabQChemInterface
   :no-value: