ACLOCAL_AMFLAGS=-I m4

AM_CXXFLAGS=@CL_CFLAGS@ @opencv_CFLAGS@ -std=c++0x -I${srcdir}/src

# Rule to turn an OpenCL source file into a C string within a header file.
# xxd uses its input's filename to name the string and its length, so we
# need to move them to a name that depends only on the path output, not its
# input.  Otherwise, builds in different relative locations would put the
# source into different variable names, and everything would fall over.
# The actual name will be 
%.h: %.cl Makefile
	cp $< $@_src
	xxd -i $@_src $@
	rm $@_src

BUILT_SOURCES = src/KeypointDetector/findMaxKernel.h

TESTS=src/test/verifyDTCWT.sh src/test/verifyInterpolation.sh
TEST_EXTENSIONS=.sh

noinst_LTLIBRARIES = libdtcwt.la libclutil.la

libdtcwt_la_SOURCES = src/DTCWT/dtcwt.cc \
					  src/DTCWT/filterer.cc \
					  src/DTCWT/fourTree.cc \
					  src/KeypointDetector/findMax.cc \
					  src/KeypointDescriptor/extractDescriptors.cc 
libdtcwt_la_LIBADD = @CL_LIBS@ @opencv_LIBS@ libclutil.la 

libclutil_la_SOURCES = src/util/clUtil.cc
libclutil_la_LIBADD = @CL_LIBS@ @opencv_LIBS@


noinst_PROGRAMS=testDTCWT testFilter verify testOutput testFindMax testRescale \
			 verifyBasic testDescriptor

testDTCWT_SOURCES = src/test/test.cc 
testDTCWT_LDADD = libdtcwt.la @opencv_LIBS@

testDescriptor_SOURCES = src/test/testDescriptor.cc 
testDescriptor_LDADD = libdtcwt.la @opencv_LIBS@

testFindMax_SOURCES = src/test/testFindMax.cc
testFindMax_LDADD = libdtcwt.la @opencv_LIBS@

testOutput_SOURCES = src/test/testOutput.cc \
					 src/MiscKernels/abs.cc
testOutput_LDADD = libdtcwt.la libclutil.la -lsfml-window -lsfml-system \
				   @opencv_LIBS@

testFilter_SOURCES = src/test/testFilter.cc 
testFilter_LDADD = libdtcwt.la @opencv_LIBS@

testRescale_SOURCES = src/test/testRescale.cc 
testRescale_LDADD = libdtcwt.la @opencv_LIBS@

verify_SOURCES = src/test/verify.cc 
verify_LDADD = libdtcwt.la @opencv_LIBS@

verifyBasic_SOURCES = src/test/verifyBasic.cc 
verifyBasic_LDADD = libdtcwt.la @opencv_LIBS@

