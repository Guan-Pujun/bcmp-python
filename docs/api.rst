API
===

AnnData workflow
----------------

.. py:currentmodule:: bcmp

.. py:function:: bcmp(adata, *, batch_key="batch", partition_n_pcs=30, partition_n_hvg=2000, k_min=3, k_max=None, batch_frac_threshold_for_k_max=0.01, min_batch_coverage=None, min_cells_in_domain_per_batch=5, max_underrepresentation_fold=10, max_residual_cell_frac=0.05, exclude_cells=None, seed=236, progress_logger=None, output_level="standard", verbose=True)

   Run the complete BCMP workflow from raw counts in ``adata.X``.
   BCMP selects highly variable genes within batches, normalizes and scales the
   counts, computes an unintegrated PCA representation, and runs the partition
   search. The input object is not modified.

   :param anndata.AnnData adata: Input object with raw counts in ``adata.X``.
      Observation and variable names must be unique.
   :param str batch_key: Column in ``adata.obs`` containing non-missing batch
      labels.
   :param int partition_n_pcs: Number of principal components used for BCMP.
   :param int partition_n_hvg: Number of highly variable genes used for BCMP.
   :param int k_min: Lower bound of the representation-scale search.
   :param int or None k_max: Upper bound of the representation-scale search. ``None``
      lets BCMP derive the bound from batch sizes.
   :param float batch_frac_threshold_for_k_max: Batches below this global
      cell fraction are ignored during automatic ``k`` upper-bound derivation.
   :param int or None min_batch_coverage: Minimum number of batches required
      in each evaluated domain. ``None`` resolves to
      ``max(2, ceil(n_batches * 0.5))``.
   :param int min_cells_in_domain_per_batch: Minimum effective cells required
      for a batch to count as represented in a domain.
   :param float max_underrepresentation_fold: Maximum allowed fold
      under-representation relative to the global batch proportion.
   :param float max_residual_cell_frac: Maximum fraction of active cells that
      may fall in residual domains excluded from the pass/fail decision.
   :param exclude_cells: Cell identifiers to exclude from partitioning. They
      remain in the returned AnnData and receive ``Excluded_<batch>`` labels.
   :type exclude_cells: sequence of str, ``pandas.Series`` or None
   :param int seed: Random seed used by preprocessing, partitioning, and debug
      UMAP.
   :param logging.Logger or None progress_logger: Optional destination for
      progress records.
   :param str output_level: ``"standard"`` (default) or ``"debug"``.
   :param bool verbose: Emit progress records when ``True``.
   :returns: An AnnData result containing a copied AnnData object, final
      selection, and search trace. Debug output adds diagnostics.

      See :doc:`output` for details.
   :rtype: bcmp.models.AnnDataResult

Embedding workflow
------------------

.. py:function:: bcmp_embedding(embedding, batch_labels, *, cell_ids=None, k_min=3, k_max=None, batch_frac_threshold_for_k_max=0.01, min_batch_coverage=None, min_cells_in_domain_per_batch=5, max_underrepresentation_fold=10, max_residual_cell_frac=0.05, seed=236, progress_logger=None, output_level="standard", verbose=True)

   Run BCMP on an existing unintegrated cell-by-dimension embedding and its
   batch labels.

   :param embedding: Cell-by-dimension embedding matrix with shape
      ``(n_cells, n_dimensions)``.
   :type embedding: numpy.ndarray
   :param batch_labels: Batch label for each row of ``embedding``. Labels must
      be non-missing and non-empty.
   :type batch_labels: sequence of str
   :param cell_ids: Unique cell identifiers. If omitted, BCMP uses
      ``"0"`` through ``"n-1"``.
   :type cell_ids: sequence of str or None
   :param int k_min: Lower bound of the representation-scale search.
   :param int or None k_max: Upper bound, or ``None`` for automatic derivation.
   :param float batch_frac_threshold_for_k_max: Small-batch threshold for
      automatic upper-bound derivation.
   :param int or None min_batch_coverage: Minimum number of batches required
      per evaluated domain.
   :param int min_cells_in_domain_per_batch: Minimum effective cells required
      for a batch to count as represented in a domain.
   :param float max_underrepresentation_fold: Maximum allowed fold
      under-representation relative to the global batch proportion.
   :param float max_residual_cell_frac: Maximum active-cell fraction assigned
      to residual domains.
   :param int seed: Random seed used by partitioning.
   :param logging.Logger or None progress_logger: Optional destination for
      progress records.
   :param str output_level: ``"standard"`` (default) or ``"debug"``.
   :param bool verbose: Emit progress records when ``True``.
   :returns: Cell-indexed labels, final selection, and search trace. Debug
      output adds diagnostics.

      See :doc:`output` for details.
   :rtype: bcmp.models.EmbeddingResult

Runtime cache
-------------

.. py:function:: set_cache_dir(cache_dir, *, force=False)

   Configure process-level cache directories for Numba, Matplotlib, and XDG
   consumers. Call this before importing cache-sensitive libraries such as
   Numba, UMAP, or Matplotlib.

   :param cache_dir: Base directory under which BCMP creates namespaced cache
      directories.
   :type cache_dir: str or pathlib.Path
   :param bool force: Override conflicting environment values when ``True``.
   :returns: The effective cache configuration.
   :rtype: dict

.. py:function:: cache_info()

   Return the currently observed process-level cache configuration. The dict
   contains ``configured``, ``source``, ``base_dir``, ``numba_cache_dir``,
   ``matplotlib_config_dir``, and ``xdg_cache_home``.

   :rtype: dict

Bundled dataset
---------------

.. py:currentmodule:: bcmp.datasets

.. py:function:: diabetic_kidney_lite()

   Load the bundled 1,500-cell diabetic kidney AnnData example used by the
   vignette. Raw counts are stored in ``X``; ``obs`` contains ``batch``.
   ``obsm["X_demo_umap"]`` contains frozen coordinates used for vignette
   visualization. Candidate integration embeddings are recorded under
   ``obsm``; their keys and display names are recorded under
   ``uns["integration_benchmark"]``.

   **Dataset source.** This bundled example is a downsampled derivative of
   the `Open Problems diabetic kidney dataset
   <https://www.openproblems.bio/datasets/cellxgene_census-dkd/>`_. The
   original study is Wilson PC, et al. *Multimodal single cell sequencing
   implicates chromatin accessibility and genetic background in diabetic kidney
   disease progression.* Nature Communications 13, 5253 (2022).
   `doi:10.1038/s41467-022-32972-z
   <https://doi.org/10.1038/s41467-022-32972-z>`_.
   The source dataset is available under the `CC BY 4.0 license
   <https://creativecommons.org/licenses/by/4.0/>`_.

   :rtype: anndata.AnnData
