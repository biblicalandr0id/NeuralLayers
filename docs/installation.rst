Installation Guide
==================

Prerequisites
-------------

* Python 3.8 or higher
* CUDA-capable GPU (optional, for acceleration)

Installation Methods
--------------------

PyPI Installation
~~~~~~~~~~~~~~~~~

The simplest way to install NeuralLayers (coming soon):

.. code-block:: bash

   pip install neurallayers

From Source
~~~~~~~~~~~

For development or to get the latest features:

.. code-block:: bash

   # Clone repository
   git clone https://github.com/biblicalandr0id/NeuralLayers.git
   cd NeuralLayers

   # Create virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt

   # Install in development mode
   pip install -e .

Docker Installation
~~~~~~~~~~~~~~~~~~~

For production deployments or reproducible environments:

.. code-block:: bash

   # Pull from Docker Hub (when available)
   docker pull neurallayers/neurallayers:latest

   # Or build locally
   docker build -t neurallayers:latest .

   # Run demo
   docker run -p 8501:8501 neurallayers:latest

Docker Compose
~~~~~~~~~~~~~~

Multiple services available:

.. code-block:: bash

   # Interactive Streamlit demo
   docker-compose up demo

   # Jupyter Lab for development
   docker-compose up jupyter
   # Access at http://localhost:8888

   # TensorBoard for training visualization
   docker-compose up tensorboard
   # Access at http://localhost:6006

   # Development environment
   docker-compose --profile dev up dev

Verify Installation
-------------------

Test your installation:

.. code-block:: bash

   python -c "import torch; from logicalbrain_network import UnifiedBrainNetwork; print('✅ Installation successful!')"

You should see: ``✅ Installation successful!``

Optional Dependencies
---------------------

Development Tools
~~~~~~~~~~~~~~~~~

For contributors and developers:

.. code-block:: bash

   pip install neurallayers[dev]

Includes:

* pytest - Testing framework
* black - Code formatter
* flake8 - Linter
* mypy - Type checker
* pre-commit - Git hooks

Jupyter Notebooks
~~~~~~~~~~~~~~~~~

For interactive exploration:

.. code-block:: bash

   pip install neurallayers[notebooks]

Includes:

* jupyter
* jupyterlab
* ipywidgets

Visualization
~~~~~~~~~~~~~

For advanced visualizations:

.. code-block:: bash

   pip install neurallayers[viz]

Includes:

* streamlit - Web apps
* plotly - Interactive plots
* seaborn - Statistical graphics

Monitoring
~~~~~~~~~~

For training monitoring:

.. code-block:: bash

   pip install neurallayers[monitoring]

Includes:

* tensorboard - Training visualization
* wandb - Experiment tracking

All Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

Install everything:

.. code-block:: bash

   pip install neurallayers[all]

Troubleshooting
---------------

CUDA Issues
~~~~~~~~~~~

If you have GPU but PyTorch doesn't detect it:

.. code-block:: bash

   # Check CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"

   # Install PyTorch with CUDA support
   pip install torch --index-url https://download.pytorch.org/whl/cu118

Import Errors
~~~~~~~~~~~~~

If you get ``ModuleNotFoundError``:

.. code-block:: bash

   # Ensure you're in the repository root
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"

   # Or install in editable mode
   pip install -e .

Memory Issues
~~~~~~~~~~~~~

For large models on limited hardware:

.. code-block:: python

   # Use smaller batch sizes
   batch_size = 8  # Instead of 32

   # Use smaller model
   model = UnifiedBrainNetwork(
       input_dim=256,
       hidden_dim=128,
       num_layers=2
   )

Platform-Specific Notes
-----------------------

Linux
~~~~~

Should work out of the box. For GPU support:

.. code-block:: bash

   # Check NVIDIA driver
   nvidia-smi

macOS
~~~~~

CPU-only support (MPS backend experimental):

.. code-block:: bash

   # Install CPU version
   pip install torch --index-url https://download.pytorch.org/whl/cpu

Windows
~~~~~~~

Use PowerShell or Command Prompt:

.. code-block:: powershell

   # Activate virtual environment (PowerShell)
   venv\Scripts\Activate.ps1

   # Or Command Prompt
   venv\Scripts\activate.bat

Next Steps
----------

After installation:

1. :doc:`quickstart` - Get started in 5 minutes
2. :doc:`tutorials/index` - Detailed tutorials
3. :doc:`examples/index` - Code examples

.. seealso::

   * `GitHub Repository <https://github.com/biblicalandr0id/NeuralLayers>`_
   * `QUICKSTART.md <../QUICKSTART.md>`_
   * `Docker Documentation <https://docs.docker.com/>`_
