$(abs_top_builddir)/packaged/lib/qat:
	mkdir -p $@

$(abs_top_builddir)/packaged/lib/qat/interop: $(abs_top_builddir)/packaged/lib/qat
	mkdir -p $@

$(abs_builddir)/qat/__init__.py: $(abs_top_builddir)/packaged/lib/qat
	touch  $@

$(abs_top_builddir)/packaged/lib/qat/interop/__init__.py:$(abs_top_builddir)/packaged/lib/qat/interop $(abs_top_builddir)/packaged/lib/qat/__init__.py
	touch  $@


# For installation
nobase_nodist_python_PYTHON=\
		  qat/interop/version.py

nobase_python_PYTHON=\
		  qat/interop/parser/gcirq2acirc.py\
		  qat/interop/parser/projectq2acirc.py\
		  qat/interop/parser/ocirc2acirc.py\
		  qat/interop/parser/oqasm2acirc.py
#File present inside build and source folder should be install with a special rule
AQASMdir=$(pythondir)/qat/interop
nodist_AQASM_PYTHON=\
		  $(srcdir)/qat/interop/__init__.py

qat/interop/version.py:${abs_top_builddir}/version.py $(abs_builddir)/qat/interop
	cp $< $@


BUILT_SOURCES=qat/interop/version.py
dist-hook:
	mkdir -p $(distdir)/qat/interop
	cp ${nodist_AQASM_PYTHON} $(distdir)/qat/interop

# For generation
all-local: $(nobase_nodist_python_PYTHON) $(abs_builddir)/qat/interop/__init__.py

clean-local:
	if test $(srcdir) != $(builddir);\
		then\
		rm -f $(builddir)/qat/interop/__init__.py;\
	fi

#
##
#
CLEANFILES=${nodist_python_PYTHON}
