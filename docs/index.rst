.. OASS documentation master file, created by
   sphinx-quickstart on Sat Jun  4 16:29:27 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to OASS!
================

Optimal Action Space Search (OASS) is an algorithm for path planning problems on directed acyclic graphs (DAG) based on reinforcement learning (RL) theory.

Installation
------------

OASS is still under development. When we are ready, you can install OASS directly via ``pip``.

.. code-block::

   pip install oass

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/UnderstandingOASS.rst
   tutorials/ShortestPath
   tutorials/AlgorithmicTrading
   tutorials/OrderExecution

.. toctree::
   :maxdepth: 1
   :caption: API Docs

   api/oass.StaticDirectedAcyclicGraph.rst
   api/oass.GradientCalculator.rst
   api/oass.AlgorithmicTrading.rst
   api/oass.ShortestPath.rst


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
