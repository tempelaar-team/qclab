.. _model:

==========================
Models
==========================

Models in QC Lab define the physics of the quantum-classical system under study. A Model object is an instance of the ``qclab.Model`` class and is equipped with a set of constants and Ingredients that specify the properties of the system in a manner that is agnostic to the quantum-classical algorithm being used.


The Model object contains a mandatory set of constants that define properties of the system:

- ``num_quantum_states``: the number of quantum states in the system,
- ``num_classical_coordinates``: the number of classical coordinates in the system,
- ``classical_coordinate_mass``: the mass of the classical coordinates,
- ``classical_coordinate_weight``: the weight of the classical coordinates (:math:`h` in the :ref:`complex-coordinate formalism <coordinates>`).


At a minimum, the Model object contains Ingredients that define the Hamiltonian of the system. QC Lab accommodates models defined in either a diabatic
(i.e. independent of the classical coordinates) or adiabatic basis. The Hamiltonian is given by three terms,

.. math::

    H(q,p) = \hat{H}_{\mathrm{q}} + \hat{H}_{\mathrm{q-c}}(q) + H_{\mathrm{c}}(q,p)

where :math:`\hat{H}_\mathrm{q}` is the quantum Hamiltonian, :math:`\hat{H}_{\mathrm{q-c}}(q)` is the quantum-classical interaction Hamiltonian, and :math:`H_{\mathrm{c}}(q,p)` is the classical Hamiltonian. These Ingredients are discussed in detail in
:ref:`Ingredients <ingredient>`. Within a diabatic basis, no other information is required to specify the Model object.

Adiabatic Basis
---------------

Within an adiabatic basis the Hamiltonian likewise consists of the same three terms of the quantum-classical Hamiltonian (where now the coordinate dependent adiabatic
potential energies are included in :math:`\hat{H}_{\mathrm{q-c}}(q)`) and a derivative coupling tensor that describes the rotation of the adiabatic basis with respect
to the classical coordinate. This tensor is given by

.. math::

    d^{\xi}_{\alpha\beta}(q) = \langle \alpha(q)\vert \frac{\partial}{\partial q_{\xi}}\vert \beta(q)\rangle

where :math:`\vert \alpha(q)\rangle` and :math:`\vert \beta(q)\rangle` are adiabatic states and :math:`\frac{\partial}{\partial q_{\xi}}` is the partial derivative with respect to the real-valued classical coordinate :math:`q_{\xi}`.
The most obvious scenario where an adiabatic basis is required is for *ab initio* models where there is no global diabatic basis. Such a Model object is implemented in
the ``qclab.models.AbInitio`` module.


The Model Class
--------------------------

The Model class is defined in the ``qclab.Model`` module. It is equipped with a Constants object ``model.constants``, an ingredients list ``model.ingredients``, and a dictionary of default constants ``model.default_constants``.


Constants
~~~~~~~~~~~~~~~~~~~~~~~~~~

Often, a model's properties can be captured by a set of high-level constants that are suitable for user input. For example, the spin-boson model is defined by the following user-defined constants:

- ``kBT``: the thermal energy at a given temperature,
- ``l_reorg``: the reorganization energy of the bath,
- ``E``: the energy bias between the two quantum states,
- ``V``: the diabatic coupling between the two quantum states,
- ``A``: the number of bosonic modes in the bath,
- ``W``: the characteristic frequency of the bosonic modes in the bath.
- ``boson_mass``: the mass of each bosonic mode in the bath.

Each of these constants have a default value stored in the dictionary ``model.default_constants``. At initialization, these defaults can be overwritten by passing a dictionary to the model constructor, as in:

.. code-block:: python

    from qclab.models import SpinBoson

    # Create a dictionary of input constants to overwrite the defaults.
    input_constants = {
        "kBT": 1.0,
        "l_reorg": 0.005,
        "E": 0.5,
        "V": 0.5,
        "A": 100,
        "W": 0.1,
        "boson_mass": 1.0
    }
    # Initialize the spin-boson model with the input constants.
    model = SpinBoson(input_constants)

These input constants are first read into the Model object's Constants object ``model.constants`` which is an instance of the ``qclab.Constants`` class. Any input constants that are not specified will take on their default values. The input constants are then used to compute the mandatory constants required by QC Lab (specified above), as well as any additional constants that may be needed by the Ingredients of the Model object. This computation is performed by a set of initialization Ingredients that are typically unique to each Model object. The resulting "internal" constants are stored in the Model object's Constants object.

For example, the spin-boson model class uses the following initialization Ingredients to compute its constants:

.. note::

    When included within a class, the first argument of the Ingredient is ``self`` instead of ``model``. Here, specifying them outside of the class, we use ``model`` to refer to the instance of the Model class.

.. code-block:: python

    def _init_h_q(model, parameters, **kwargs):
        """
        Initializes the constants required for the two-level quantum Hamiltonian.
        """
        model.constants.two_level_00 = model.constants.get("E")
        model.constants.two_level_11 = -model.constants.get("E")
        model.constants.two_level_01_re = model.constants.get("V")
        model.constants.two_level_01_im = 0
        return

    def _init_h_qc(model, parameters, **kwargs):
        """
        Initializes the constants required for the diagonal linear quantum-classical Hamiltonian.
        """
        A = model.constants.get("A")
        l_reorg = model.constants.get("l_reorg")
        boson_mass = model.constants.get("boson_mass")
        h = model.constants.classical_coordinate_weight
        w = model.constants.harmonic_frequency
        model.constants.diagonal_linear_coupling = np.zeros((2, A))
        model.constants.diagonal_linear_coupling[0] = (
            w * np.sqrt(2.0 * l_reorg / A) * (1.0 / np.sqrt(2.0 * boson_mass * h))
        )
        model.constants.diagonal_linear_coupling[1] = (
            -w * np.sqrt(2.0 * l_reorg / A) * (1.0 / np.sqrt(2.0 * boson_mass * h))
        )
        return

    def _init_h_c(model, parameters, **kwargs):
        """
        Initializes the constants required for the harmonic classical Hamiltonian.
        """
        A = model.constants.get("A")
        W = model.constants.get("W")
        model.constants.harmonic_frequency = W * np.tan(
            np.arange(0.5, A + 0.5, 1.0) * np.pi * 0.5 / A
        )
        return

    def _init_model(model, parameters, **kwargs):
        """
        Initializes the mandatory constants required by QC Lab.
        """
        A = model.constants.get("A")
        boson_mass = model.constants.get("boson_mass")
        model.constants.num_classical_coordinates = A
        model.constants.num_quantum_states = 2
        model.constants.classical_coordinate_weight = model.constants.harmonic_frequency
        model.constants.classical_coordinate_mass = boson_mass * np.ones(A)
        return


For more information on the formatting of an Ingredient, please refer to :ref:`Ingredients <ingredient>`. In the subsequent section we will discuss how these Ingredients are included in a Model class.


Ingredients List
~~~~~~~~~~~~~~~~~~~~~~~~~~

The Ingredients in a Model object are contained in a list of tuples ``model.ingredients``. Each tuple contains the name of the Ingredient as a string and the Ingredient function itself. For example, the spin-boson model includes the following Ingredients:


.. code-block:: python

    ingredients = [
        ("h_q", ingredients.h_q_two_level),
        ("h_qc", ingredients.h_qc_diagonal_linear),
        ("h_c", ingredients.h_c_harmonic),
        ("dh_qc_dzc", ingredients.dh_qc_dzc_diagonal_linear),
        ("dh_c_dzc", ingredients.dh_c_dzc_harmonic),
        ("init_classical", ingredients.init_classical_boltzmann_harmonic),
        ("hop", ingredients.hop_harmonic),
        ("_init_h_q", _init_h_q),
        ("_init_h_qc", _init_h_qc),
        ("_init_model", _init_model),
        ("_init_h_c", _init_h_c),
    ]


The ingredients list includes the Hamiltonian Ingredients (``h_q``, ``h_qc``, ``h_c``), their gradients (``dh_qc_dzc``, ``dh_c_dzc``), and other Ingredients used in the dynamics (``init_classical``, ``hop``). Other Ingredients define initialization steps that compute the Model object's constants (``_init_h_q``, ``_init_h_qc``, ``_init_h_c``, ``_init_model``). These Ingredients are distinguished by their leading underscore, which indicates that they are to be run when the Model object is initialized.

To initialize the Model object's constants manually one can run

.. code-block:: python

    model.initialize_constants()

which will execute all the Ingredients in the list that begin with an underscore. After doing so, all the internal constants will be available in the Model object's Constants object ``model.constants``. By default, this is done whenever a Model object is initialized and whenever a constant is changed.


The ``model.ingredients`` list is unordered in principle — each slot is looked up back-to-front, so the last entry registered under a given slot name takes precedence. This makes it possible to override an existing Ingredient by appending a new tuple to the list. For example, if we wanted to change the quantum-classical interaction from diagonal to off-diagonal coupling, we could define a new Ingredient and append it to the ingredients list:

.. code-block:: python

    from qclab import ingredients

    def h_qc_off_diagonal(model, parameters, **kwargs):
        """
        A vectorized Ingredient that couples the boson coordinates
        to the off-diagonal elements of the quantum Hamiltonian.
        """
        z = kwargs['z']
        A = model.constants.get("A")
        m = model.constants.classical_coordinate_mass
        h = model.constants.classical_coordinate_weight
        g = model.constants.diagonal_linear_coupling[0]
        N = model.constants.num_quantum_states
        batch_size = len(z)
        h_qc = np.zeros((batch_size, N, N), dtype=complex)
        h_qc[:, 0, 1] = g[np.newaxis, :] * (z + np.conj(z))
        h_qc[:, 1, 0] = np.conj(h_qc[:, 0, 1])
        return h_qc

    # Overwrite the quantum-classical interaction Hamiltonian Ingredient.
    model.ingredients.append(("h_qc", h_qc_off_diagonal))
    # Overwrite the gradient of the quantum-classical interaction Hamiltonian Ingredient.
    model.ingredients.append(("dh_qc_dzc", None))  # No analytical gradient available.


.. _spinboson_model:

Spin-Boson Model
--------------------------

.. list-table:: Spin-Boson Model Constants
   :header-rows: 1
   :widths: 25 50 25

   * - Constant
     - Description
     - Default
   * - ``kBT``
     - Thermal energy.
     - 1.0
   * - ``V``
     - Onsite energy.
     - 0.5
   * - ``E``
     - Diabatic coupling.
     - 0.5
   * - ``A``
     - Number of bosonic modes.
     - 100
   * - ``W``
     - Characteristic frequency.
     - 0.1
   * - ``l_reorg``
     - Reorganization energy.
     - 0.005
   * - ``boson_mass``
     - Boson mass.
     - 1.0


.. dropdown:: View full source
   :icon: code

   .. literalinclude:: ../../src/qclab/models/spin_boson.py
      :language: python
      :linenos:

FMO Complex Model
--------------------------

.. list-table:: FMO Model Constants
   :header-rows: 1
   :widths: 25 50 25

   * - Constant
     - Description
     - Default
   * - ``kBT``
     - Thermal energy.
     - 1
   * - ``mass``
     - Coordinate mass.
     - 1
   * - ``l_reorg``
     - Reorganization energy.
     - 35 cm :sup:`-1`
   * - ``w_c``
     - Characteristic frequency.
     - 106.14 cm :sup:`-1`
   * - ``N``
     - Number of bosonic modes.
     - 200


.. dropdown:: View full source
   :icon: code

   .. literalinclude:: ../../src/qclab/models/fmo_complex.py
      :language: python
      :linenos:


Tully Problem One
--------------------------

.. list-table:: Tully Problem One Model Constants
   :header-rows: 1
   :widths: 25 50 25

   * - Constant
     - Description
     - Default
   * - ``init_momentum``
     - Initial momentum.
     - 10.0
   * - ``init_position``
     - Initial position.
     - -25.0
   * - ``mass``
     - Coordinate mass.
     - 2000.0
   * - ``A``
     - See reference publication.
     - 0.01
   * - ``B``
     - See reference publication.
     - 1.6
   * - ``C``
     - See reference publication.
     - 0.005
   * - ``D``
     - See reference publication.
     - 1.0

.. dropdown:: View full source
   :icon: code

   .. literalinclude:: ../../src/qclab/models/tully_problem_one.py
      :language: python
      :linenos:


