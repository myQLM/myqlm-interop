
if HAVE_TESTS
all-local:
	mkdir -p report

#TESTS_ENVIRONMENT=@VALGRIND@ @VALGRIND_ARGS@
AUTOMAKE_OPTIONS = parallel-tests
TEST_EXTENSIONS = .pl .sh .py
LOG_COMPILER =${VALGRIND}  `if   test "${VALGRIND}" !=  ""   ; then echo "${VALGRIND_ARGS}"; fi`
LOG_DRIVER=$(top_srcdir)/custom-test-driver
PY_LOG_DRIVER=PYTHONPATH=${abs_top_srcdir}/packaged/lib/:${abs_top_builddir}/packaged/lib/:${abs_top_srcdir}/tests:$${PYTHONPATH} 			   PATH=${abs_top_srcdir}/packaged/bin:${abs_top_srcdir}/packaged/doc/examples:${PATH}			   LD_LIBRARY_PATH=${abs_top_builddir}/packaged/lib/.libs:$${LD_LIBRARY_PATH} 			   ${LOG_DRIVER}

compile_tests:$(check_PROGRAMS)


force:${top_srcdir}/valgrind.supp
	mkdir -p report
	-[ -f $$(basename $^)  ] || cp $^ .

DISTCLEANFILES=			   valgrind.supp			   report/${PACKAGE_NAME}-Results.xml			   report/${PACKAGE_NAME}-Listing.xml


endif
            