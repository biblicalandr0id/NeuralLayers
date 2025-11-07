NeuralLayers Documentation
===========================

.. image:: https://img.shields.io/badge/license-Proprietary-red.svg
   :target: ../LICENSE.txt
   :alt: License

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python

.. image:: https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg
   :target: https://pytorch.org/
   :alt: PyTorch

Welcome to NeuralLayers!
------------------------

**NeuralLayers** is a production-ready framework for neural-logical computing that bridges biological brain simulation, formal logical reasoning, and consciousness-like abstractions.

Key Features
~~~~~~~~~~~~

* 🧠 **Biological Realism**: Hodgkin-Huxley dynamics, calcium signaling, ATP metabolism
* 🔬 **Logical Reasoning**: Explicit rules with Fibonacci weighting and golden ratio activation
* 🌌 **Consciousness Modeling**: 7-layer hierarchical consciousness framework
* 🏗️ **Unified Architecture**: Cerebrum, Cerebellum, Brainstem modules integrated
* ⚡ **Production Ready**: Docker, CI/CD, benchmarks, comprehensive tests

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install neurallayers

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import torch
   from logicalbrain_network import UnifiedBrainNetwork

   # Create model
   model = UnifiedBrainNetwork(
       input_dim=1024,
       hidden_dim=512,
       num_layers=4
   )

   # Forward pass
   x = torch.randn(16, 1024)
   output = model(x)

   # Access system state
   membrane_potential = output['system_state']['V']
   truth_values = output['system_state']['tau']

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   tutorials/index
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: Core Concepts

   concepts/architecture
   concepts/brain_modules
   concepts/logical_reasoning
   concepts/consciousness
   concepts/system_state

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/logicalbrain_network
   api/brain_network
   api/logical_reasoning
   api/consciousness_layers
   api/umi_layer
   api/utils

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   advanced/training
   advanced/benchmarking
   advanced/deployment
   advanced/optimization

.. toctree::
   :maxdepth: 2
   :caption: Research

   research/mathematical_foundations
   research/biophysical_modeling
   research/consciousness_theory
   research/experiments

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog
   license

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Links
-----

* `GitHub Repository <https://github.com/biblicalandr0id/NeuralLayers>`_
* `Issue Tracker <https://github.com/biblicalandr0id/NeuralLayers/issues>`_
* `Quick Start Guide <../QUICKSTART.md>`_
* `Contributing Guide <../CONTRIBUTING.md>`_
