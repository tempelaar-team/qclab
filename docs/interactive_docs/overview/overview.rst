.. _overview:

====================
QC Lab Architecture
====================

We prefer to think of running a quantum-classical simulation as like preparing a delicious meal. 
In that analogy, the Simulation object is equipped with a Model object that provides "ingredients", 
an Algorithm object that provides the "recipe". The Simulation object is then submitted to a dynamics driver that executes the recipe step by step.
The Data object is the output dish that results from the simulation.

A key feature of this architecture is that different models and algorithms can be swapped in and out with no 
modification to the rest of the simulation. This enables users to easily experiment with different
quantum-classical simulation algorithms and models, allowing for rapid prototyping and testing of new ideas.

This construction is depicted diagrammatically below. Click on any component to learn more about it!


.. container:: graphviz-center

   .. graphviz::

      digraph flow {
        rankdir=TB;
        bgcolor="transparent";

        node [
          fontsize=12
          fontname="Helvetica, Arial, sans-serif"
          margin="0.3,0.2"
          style=filled
          fillcolor=white
          color="#f38c3c"
        ];

        sim   [label="Simulation Object", URL="../../user_guide/simulation.html"];
        model [label="Model Object",      URL="../../user_guide/model.html"];
        algo  [label="Algorithm Object",  URL="../../user_guide/algorithm.html"];
        driver[label="Dynamics Driver",   URL="../../user_guide/driver.html"];
        data  [label="Data Object",       URL="../../user_guide/data.html"];
        ingredients [label="Ingredients", URL="../../user_guide/ingredient.html"];
        tasks [label="Tasks", URL="../../user_guide/task.html"];

        ingredients -> model [color="#f38c3c"];
        tasks -> algo [color="#f38c3c"];
        model -> sim [color="#f38c3c"];
        algo  -> sim [color="#f38c3c"];
        sim   -> driver [color="#f38c3c"];
        driver-> data [color="#f38c3c"];
      }

