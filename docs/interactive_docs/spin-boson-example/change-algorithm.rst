.. _change-algorithm:

.. I want to use FSSH instead.
.. ===========================

Sure! Following the last example you can simply swap out ``sim.algorithm`` to use ``FewestSwitchesSurfaceHopping``.

.. code-block:: python


    from qclab.algorithms import FewestSwitchesSurfaceHopping

    sim.algorithm = FewestSwitchesSurfaceHopping()

The output has changed once more:


.. image:: fssh_lreorg.png
    :alt: Population dynamics.
    :align: center
    :width: 50%


You can learn about algorithms in the :ref:`Algorithms <algorithm>` documentation.


.. note::

    The populations above are not in agreement at the outset of the simulation because the FSSH algorithm 
    stochastically samples the initial state while the mean-field algorithm does not. If the number of trajectories 
    were increased, the initial populations would converge to the same value.
