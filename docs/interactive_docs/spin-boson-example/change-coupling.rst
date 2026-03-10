.. _change-coupling:


I want to switch to off-diagonal coupling.
==========================================


Changing the coupling term is straightforward! We'll make a new ingredient that couples the boson coordinates to the off-diagonal
elements of the quantum Hamiltonian.

For simplicity we will make an ingredient that creates the quantum-classical term of the Hamiltonian for a single trajectory and then use 
QC Lab's built-in vectorization decorator to automatically vectorize it.


.. code-block:: python

    from qclab.ingredients import vectorize_ingredient

    @vectorize_ingredient
    def h_qc(model, parameters, **kwargs):
        """
        A coupling term that couples the boson coordinates to the off-diagonal elements of the quantum Hamiltonian.
        """
        # First we'll get the z coordinate from the keyword arguments
        z = kwargs['z']
        # Next we'll get the Required constants from the constants object.
        m = model.constants.classical_coordinate_mass
        h = model.constants.classical_coordinate_weight
        g = model.constants.harmonic_frequency * np.sqrt(2 * model.constants.l_reorg / model.constants.A)
        # Now we can construct the empty Hamiltonian matrix as a 2x2 complex array.
        h_qc = np.zeros((2, 2), dtype=complex)
        # Then we can populate the off-diagonal elements of the Hamiltonian matrix.
        h_qc[0, 1] = np.sum((g * np.sqrt(1 / (2 * m * h))) * (z + np.conj(z)))
        h_qc[1, 0] = np.conj(h_qc[0, 1])
        return h_qc

Next we can add the ingredient to the model's ingredients list, and overwrite the analytical gradient ingredient
which is no longer correct for the new coupling. QC Lab will automatically differentiate the new coupling term 
using finite differences.

.. code-block:: python


    # Add the new coupling term to the model's ingredients.
    sim.model.ingredients.append(("h_qc", h_qc))
    # Overwrite the analytical gradient ingredient, which is no longer correct for the new coupling.
    sim.model.ingredients.append(("dh_qc_dzc", None))


Now we can run the simulation with the new coupling term and compare the results to the previous simulation. 
You'll notice a small decrease in the performance of the simulation due to the numerical calculation of the gradients. 
If you'd like to speed up the simulation, you can implement an analytical gradient for the new coupling term by following the 
`model development guide <../../developer_guide/model_dev/model_dev.html>`_.


.. image:: fssh_lreorg_inv_vel_offdiag.png
    :alt: Population dynamics.
    :align: center
    :width: 50%
