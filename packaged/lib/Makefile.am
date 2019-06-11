$(abs_top_builddir)/packaged/lib/qat:
	mkdir -p $@

$(abs_top_builddir)/packaged/lib/qat/interop: $(abs_top_builddir)/packaged/lib/qat
	mkdir -p $@

$(abs_builddir)/qat/__init__.py: $(abs_top_builddir)/packaged/lib/qat
	touch  $@

# For installation
nobase_nodist_python_PYTHON=\
		  qat/interop/version.py

nobase_python_PYTHON=\
	qat/interop/__init__.so \
	qat/interop/cirq/converters.so \
	qat/interop/openqasm/oqasm_routine.so \
	qat/interop/openqasm/qasm_lexer.so \
	qat/interop/openqasm/qasm_parser.so \
	qat/interop/projectq/converters.so \
	qat/interop/pyquil/algorithms.so \
	qat/interop/pyquil/converters.so \
	qat/interop/pyquil/providers.so \
	qat/interop/qiskit/algorithms.so \
	qat/interop/qiskit/converters.so \
	qat/interop/qiskit/providers.so

#File present inside build and source folder should be install with a special rule
AQASMdir=$(pythondir)/qat/interop
nodist_AQASM_PYTHON=\
		  $(srcdir)/qat/interop/__init__.py

qat/interop/version.py:${abs_top_builddir}/version.py $(abs_builddir)/qat/interop
	cp $< $@

# Cythonization rule
%.so: %.py
	mkdir -p $$(dirname $@)
	cython -3 -Werror -Wextra $< -o $*.c
	gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I$$(${PYTHON}-config --includes) -o $@ $*.c

BUILT_SOURCES=\
	qat/interop/version.py \
	${nobase_python_PYTHON}

dist-hook:
	mkdir -p $(distdir)/qat/interop
	cp ${nodist_AQASM_PYTHON} $(distdir)/qat/interop

# For generation
all-local: $(nobase_nodist_python_PYTHON) \
		   $(abs_builddir)/qat/interop/__init__.py

clean-local:
	if test $(srcdir) != $(builddir);\
		then\
		rm -f $(builddir)/qat/interop/__init__.py;\
	fi

#
##
#
CLEANFILES=${nodist_python_PYTHON}
