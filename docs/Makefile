# Minimal makefile for Sphinx documentation.
#
#   make html        build docs/_build/html (open docs/_build/html/index.html)
#   make strict      the same, but any warning fails the build (what Read the Docs will do once
#                    fail_on_warning is on in .readthedocs.yaml)
#   make linkcheck   check every external link
#   make clean       remove docs/_build and the generated api/ and tools/ pages
#
# You can set these variables from the command line, and also from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help strict clean Makefile

strict:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) -W --keep-going

clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	rm -rf "$(SOURCEDIR)/api" "$(SOURCEDIR)/tools"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
