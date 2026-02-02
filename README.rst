QC Lab: A package for quantum-classical modeling
================================================


**QC Lab** is a Python package designed for implementing and executing quantum-classical (QC) dynamics simulations.
It offers an environment for developing physical models and QC algorithms which enables algorithms and models to be combined arbitrarily.
QC Lab comes with a variety of already implemented models and algorithms which we hope will encourage new researchers to explore the field of quantum-classical dynamics. Users that implement their own models and algorithms will have the opportunity to contribute them to QC Lab to form a growing library of quantum-classical dynamics tools.

As of QC Lab v1.1.0, QC Lab comes with the ability to simulate on-the-fly ab initio nonadiabatic dynamics. 


**QC Lab** is developed and maintained by the Tempelaar Team in the Chemistry Department of Northwestern University in Evanston, Illinois, USA.


The documentation for QC Lab can be found at https://tempelaar-team.github.io/qclab/.


Capabilities
------------

Dynamics Algorithms
```````````````````

The following algorithms are implemented making use of the complex-classical coordinate formalism established in [1].


* Mean-field (Ehrenfest) dynamics [2]
* Fewest-switches surface hopping (FSSH) dynamics [3]
* Ab Initio Mean-Field dynamics
* Ab Initio FSSH dynamics

Model Systems
`````````````

* Spin-boson model [4]
* Holstein lattice model [5]
* Fenna-Matthews-Olson (FMO) complex [6, 7]
* Tully problems I, II, & III [8]
* Atomistic Ab Initio model


Installing QC Lab
-----------------

QC Lab can be installed from the Python Package Index (PyPI) by executing::

   pip install qclab

To install QC Lab without h5py, numba, or ASE support, execute::

   pip install qclab --no-deps
   pip install numpy tqdm

to install the remaining required dependencies manually.

QC Lab can be installed from source by downloading the `latest release <https://github.com/tempelaar-team/qclab/releases>`_,
unpacking it, and executing::

   pip install ./

from inside its topmost directory (where the `pyproject.toml` file is located).


QC Lab doesn’t enforce third-party dependencies. If you hit resolver conflicts or install errors, the quickest fix is to install in a clean Python environment (via `venv` or `conda`). Alternatively, reconcile package versions in your existing environment until the requirements are satisfied.


Bibliography
------------

1. Miyazaki, K.; Krotz, A.; Tempelaar, R. Mixed Quantum–Classical Dynamics under Arbitrary Unitary Basis Transformations. J. Chem. Theory Comput. 2024, 20 (15), 6500–6509. https://doi.org/10.1021/acs.jctc.4c00555.
2. Tully, J. C. Mixed Quantum–Classical Dynamics. Faraday Discuss. 1998, 110 (0), 407–419. https://doi.org/10.1039/A801824C.
3. Hammes‐Schiffer, S.; Tully, J. C. Proton Transfer in Solution: Molecular Dynamics with Quantum Transitions. J. Chem. Phys. 1994, 101 (6), 4657–4667. https://doi.org/10.1063/1.467455.
4. Tempelaar, R.; Reichman, D. R. Generalization of Fewest-Switches Surface Hopping for Coherences. J. Chem. Phys. 2018, 148 (10), 102309. https://doi.org/10.1063/1.5000843.
5. Krotz, A.; Provazza, J.; Tempelaar, R. A Reciprocal-Space Formulation of Mixed Quantum–Classical Dynamics. J. Chem. Phys. 2021, 154 (22), 224101. https://doi.org/10.1063/5.0053177.
6. Fenna, R. E.; Matthews, B. W. Chlorophyll Arrangement in a Bacteriochlorophyll Protein from Chlorobium Limicola. Nature 1975, 258 (5536), 573–577. https://doi.org/10.1038/258573a0.
7. Mulvihill, E.; Lenn, K. M.; Gao, X.; Schubert, A.; Dunietz, B. D.; Geva, E. Simulating Energy Transfer Dynamics in the Fenna–Matthews–Olson Complex via the Modified Generalized Quantum Master Equation. J. Chem. Phys. 2021, 154 (20), 204109. https://doi.org/10.1063/5.0051101.
8. Tully, J. C. Molecular Dynamics with Electronic Transitions. J. Chem. Phys. 1990, 93 (2), 1061–1071. https://doi.org/10.1063/1.459170.
