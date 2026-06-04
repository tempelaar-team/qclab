.. _modify-fssh:


.. Modifying the FSSH Algorithm
.. ============================

Let's try modifying the FSSH Algorithm object so that the velocities of frustrated trajectories are reversed.
In the :ref:`complex coordinate formalism <coordinates>`, this means conjugating the `z` coordinate of the frustrated trajectories.
To this end, we write the following function:

.. code-block:: python


    def update_z_reverse_frustrated_fssh(sim, state, parameters):
        """
        Reverse the velocities of frustrated trajectories in the FSSH algorithm.
        """
        # Get the indices of trajectories that were frustrated
        # (i.e., did not successfully hop but were eligible to hop).
        frustrated_indices = state["hop_ind"][~state["hop_successful"]]
        # Reverse the velocities for these indices, in the complex classical coordinate
        # formalism, this means conjugating the z coordinate.
        state["z"][frustrated_indices] = state["z"][frustrated_indices].conj()
        return state, parameters


Now we can insert this function as a Task into an instance of the FSSH Algorithm object. To know where we should insert it, we can look
at the ``update_recipe`` of the FSSH Algorithm object (see :ref:`fssh_source`).

A good place to invert the velocities of frustrated trajectories is just at the end of the active surface updates.
We append the new Task to the end of the update Recipe using a standard Python list method:

.. code-block:: python

    # Insert the function for reversing velocities as a task into the update recipe.
    sim.algorithm.update_recipe.append(update_z_reverse_frustrated_fssh)


The output has now changed to:


.. image:: fssh_lreorg_inv_vel.png
   :alt: Modified FSSH populations.
   :align: center
   :width: 50%