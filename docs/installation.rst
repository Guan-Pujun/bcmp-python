Installation
============

Compatibility
-------------

* Version: |release| (Release Date: September 2026)
* Tested Python versions: 3.12–3.14
* Tested platforms: Linux, Windows, and macOS on Apple Silicon

Linux and Windows
-----------------

Install from PyPI:

.. code-block:: console

   python -m pip install bcmp

macOS (Apple Silicon)
---------------------

Use the Conda-forge environment. Download :download:`bcmp.yml <../envs/bcmp.yml>`
and run:

.. code-block:: console

   conda env create -f bcmp.yml
   conda activate bcmp
   python -m pip install --no-deps bcmp

Verify the installation
-----------------------

.. code-block:: python

   import bcmp
   from bcmp.datasets import diabetic_kidney_lite

   adata = diabetic_kidney_lite()
   print(adata.shape)

Expected output:

.. code-block:: text

   (1500, 27980)

Install from source
-------------------

For the development version:

.. code-block:: console

   git clone https://github.com/Guan-Pujun/bcmp-python.git
   cd bcmp-python
   python -m pip install .

scIB metric calculation
-----------------------

Batch-effect-removal metrics for candidate integration embeddings are
calculated with ``scib-metrics``. Follow the `official scib-metrics installation guide
<https://scib-metrics.readthedocs.io/en/stable/#installation>`_; for the PyPI
release, run:

.. code-block:: console

   python -m pip install scib-metrics

Dependencies
------------

Installing ``bcmp`` resolves and installs the following core packages:

* ``anndata``
* ``numpy``
* ``pandas``
* ``scipy``
* ``numba``
* ``annoy``
* ``igraph`` and ``leidenalg``
* ``scikit-learn`` and ``scikit-misc``
* ``umap-learn``

For typical use, we recommend:

* ``scib-metrics``
