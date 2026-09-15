Output
======

Output levels
-------------

``output_level="standard"`` is the default. ``debug`` keeps the same final
result and search trace, then adds the information needed to inspect individual
candidate partitions.

.. role:: bcmp-debug
   :class: bcmp-debug

.. container:: bcmp-output-level-grid

   .. container:: bcmp-output-level-card

      :ref:`Standard (default) <output-standard>`

      .. code-block:: text

         result
         ├── labels / adata
         ├── selection
         └── search_trace

   .. container:: bcmp-output-level-card bcmp-output-level-card--debug

      :ref:`Debug <output-debug>`

      .. parsed-literal::

         result
         ├── labels / adata
         ├── selection
         ├── search_trace
         └── :bcmp-debug:`debug`

.. _output-standard:

Standard output
---------------

Domain labels
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 36 40

   * - Entry point
     - Result location
     - Meaning
   * - ``bcmp(...)``
     - ``result.adata.obs["bcmp_domain"]``
     - BCMP domain labels in a copied AnnData object. The input AnnData is not modified.
   * - ``bcmp_embedding(...)``
     - ``result.labels``
     - A cell-indexed series of BCMP domain labels.

Final selection
~~~~~~~~~~~~~~~

``result.selection`` describes the final partition:

.. list-table::
   :header-rows: 1
   :widths: 33 67

   * - Name
     - Meaning
   * - ``selected_k``
     - Representation scale selected by the search.
   * - ``n_domains``
     - Number of domains in the selected partition.
   * - ``mixing_status``
     - ``"pass"`` or ``"fail"`` for the selected partition.
   * - ``n_evaluated_domains``
     - Domains used in the mixing decision.
   * - ``n_residual_domains`` / ``n_residual_cells``
     - Residual domains and their active cells.
   * - ``residual_cell_frac``
     - Fraction of active cells in residual domains.

Search trace
~~~~~~~~~~~~

``result.search_trace`` records the ``k`` values BCMP tested to choose
``selected_k``; each row is one tested ``k``:

.. list-table::
   :header-rows: 1
   :widths: 27 73

   * - Name
     - Meaning
   * - ``iter_id`` / ``phase`` / ``k``
     - Evaluation order, role (``k_low``, ``k_high``, or ``k_mid``), and scale.
   * - ``mixing_status``
     - Mixing outcome at that scale.
   * - ``n_domains`` / ``n_failed_domains`` / ``n_residual_domains``
     - Domain counts at that scale.
   * - ``actual_residual_cell_frac``
     - Observed active-cell fraction in residual domains.

.. _output-debug:

Debug output
------------

Debug output is for examining how the search arrived at its result. It retains
the standard fields and adds ``result.debug``:

.. parsed-literal::

   result
   ├── labels / adata
   │           └── :bcmp-debug:`obsm`
   │               ├── :bcmp-debug:`X_bcmp_pca`
   │               └── :bcmp-debug:`X_bcmp_umap`
   ├── selection
   ├── search_trace
   └── :bcmp-debug:`debug`
       ├── :bcmp-debug:`parameters`
       ├── :bcmp-debug:`search`
       ├── :bcmp-debug:`candidates[k]`
       │   ├── :bcmp-debug:`partition`
       │   └── :bcmp-debug:`domain_summary`
       └── :bcmp-debug:`workflow`
           ├── :bcmp-debug:`hvg_genes`
           └── :bcmp-debug:`excluded_cell_ids`

The ``parameters`` field records the input values used for the run; ``search``
records the resolved ``k`` bounds and search outcome. Each ``candidates[k]``
entry corresponds to one row of ``search_trace``.
``partition`` is a cell-indexed domain-label series, while ``domain_summary``
reports domain size, cell fraction, batch support, and mixing status.

In debug output from ``bcmp(...)``, the additional ``workflow`` contains the selected HVG genes and
excluded cell IDs; ``adata.obsm`` additionally contains the uncorrected PCA in
``X_bcmp_pca`` and a diagnostic UMAP in ``X_bcmp_umap``.
