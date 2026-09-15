SPHINXBUILD ?= sphinx-build
ROOTDIR := $(abspath ..)
SOURCEDIR = $(ROOTDIR)/docs
BUILDDIR ?= $(ROOTDIR)/analyses/tmp/manual_docs
CACHEDIR ?= $(ROOTDIR)/analyses/tmp/manual_docs_cache

.PHONY: html clean

html:
	NUMBA_CACHE_DIR="$(CACHEDIR)/numba" \
	MPLCONFIGDIR="$(CACHEDIR)/matplotlib" \
	XDG_CACHE_HOME="$(CACHEDIR)/xdg" \
	PYTHONPATH="$(ROOTDIR)/src" \
	$(SPHINXBUILD) -W --keep-going -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"

clean:
	@echo "Generated documentation is kept under $(BUILDDIR); move it to TRASH manually if no longer needed."
