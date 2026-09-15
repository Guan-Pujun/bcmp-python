Method
======

Batch-Constraint Manifold Partition (BCMP) derives proxy population labels
directly from unintegrated single-cell data geometry. These labels can be used
to compare batch-effect removal across candidate integration methods
before cell annotations are available.

How BCMP works
--------------

.. raw:: html

   <ol class="bcmp-method-flow">
     <li>
       <strong>Represent the unintegrated manifold</strong>
       <span>Use the PCA representation constructed from raw counts.</span>
     </li>
     <li>
       <strong>Partition at candidate representation scales</strong>
       <span>For each candidate representation scale <code>k</code>, build a cell-cell graph and use community detection to define candidate domains.</span>
     </li>
     <li>
       <strong>Evaluate multi-batch support</strong>
       <span>At each candidate scale, determine whether every evaluated domain has sufficient support from multiple batches.</span>
     </li>
     <li>
       <strong>Select the smallest passing <code>k</code></strong>
       <span>Return the candidate domains from the smallest scale that passes the multi-batch support criterion.</span>
     </li>
   </ol>

.. only:: latex

   #. **Represent the unintegrated manifold.** Use the PCA representation
      constructed from raw counts.

   #. **Partition at candidate representation scales.** For each candidate
      representation scale ``k``, build a cell-cell graph and use community
      detection to define candidate domains.

   #. **Evaluate multi-batch support.** At each candidate scale, determine
      whether every evaluated domain has sufficient support from multiple
      batches.

   #. **Select the smallest passing ``k``.** Return the candidate domains from
      the smallest scale that passes the multi-batch support criterion.

Choose an implementation
------------------------

BCMP is available in Python and R. Choose the package that fits the language
and data structure already used in your analysis.

.. raw:: html

   <div class="bcmp-choice-grid">
     <article class="bcmp-choice-card bcmp-choice-card--py">
       <div class="bcmp-choice-card__header">
         <span class="bcmp-choice-card__logo">
           <img class="only-light" src="_static/bcmp-python-light.svg" alt="BCMP Py package logo">
           <img class="only-dark pst-js-only" src="_static/bcmp-python-dark.svg" alt="BCMP Py package logo">
         </span>
         <div>
           <h2>BCMP<sup>Py</sup></h2>
           <p>Python package</p>
         </div>
       </div>
       <p>Use an <code>AnnData</code> object, or supply an
       existing unintegrated cell-by-dimension embedding matrix.</p>
       <a class="bcmp-choice-card__action" href="installation.html">Open the Python manual</a>
     </article>
     <article class="bcmp-choice-card bcmp-choice-card--r">
       <div class="bcmp-choice-card__header">
         <span class="bcmp-choice-card__logo">
           <img class="only-light" src="_static/bcmp-r-light.svg" alt="BCMP R package logo">
           <img class="only-dark pst-js-only" src="_static/bcmp-r-dark.svg" alt="BCMP R package logo">
         </span>
         <div>
           <h2>BCMP<sup>R</sup></h2>
           <p>R package</p>
         </div>
       </div>
       <p>Use a <code>Seurat</code> object, a
       <code>SingleCellExperiment</code> object, or supply an existing
       unintegrated cell-by-dimension embedding matrix.</p>
       <a class="bcmp-choice-card__action" href="https://guan-pujun.github.io/bcmp-r/">Open the R manual</a>
     </article>
   </div>

.. only:: latex

   **BCMP Py** -- **Python package**

      Use an ``AnnData`` object, or supply an existing unintegrated
      cell-by-dimension embedding matrix. See :doc:`Installation
      <installation>`.

   **BCMP R** -- **R package**

      Use a ``Seurat`` object, a ``SingleCellExperiment`` object, or supply an
      existing unintegrated cell-by-dimension embedding matrix. The R manual
      will be added with the public release.
