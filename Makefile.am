ACLOCAL_AMFLAGS=-I m4

AM_CXXFLAGS=@CL_CFLAGS@ @opencv_CFLAGS@ -std=c++0x

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

TESTS=test/verifyDTCWT.sh test/verifyInterpolation.sh
TEST_EXTENSIONS=.sh

bin_PROGRAMS=testDTCWT testFilter verify testOutput testFindMax testRescale \
			 verifyBasic testDescriptor

testDTCWT_SOURCES=test.cc filterer.cc clUtil.cc dtcwt.cc extractDescriptors.cc
testDTCWT_LDADD=@CL_LIBS@ @opencv_LIBS@

testDescriptor_SOURCES=testDescriptor.cc \
					   filterer.cc clUtil.cc dtcwt.cc extractDescriptors.cc
testDescriptor_LDADD=@CL_LIBS@ @opencv_LIBS@

testFindMax_SOURCES=testFindMax.cc filterer.cc clUtil.cc dtcwt.cc findMax.cc
testFindMax_LDADD=@CL_LIBS@ @opencv_LIBS@

testOutput_SOURCES=testOutput.cc filterer.cc clUtil.cc dtcwt.cc findMax.cc \
				   abs.cc
testOutput_LDADD=@CL_LIBS@ @opencv_LIBS@ -lsfml-window -lsfml-system

testFilter_SOURCES=testFilter.cc filterer.cc clUtil.cc dtcwt.cc
testFilter_LDADD=@CL_LIBS@ @opencv_LIBS@

testRescale_SOURCES=testRescale.cc filterer.cc clUtil.cc dtcwt.cc
testRescale_LDADD=@CL_LIBS@ @opencv_LIBS@

verify_SOURCES=verify.cc filterer.cc clUtil.cc dtcwt.cc
verify_LDADD=@CL_LIBS@ @opencv_LIBS@

verifyBasic_SOURCES=test/verifyBasic.cc filterer.cc clUtil.cc dtcwt.cc
verifyBasic_LDADD=@CL_LIBS@ @opencv_LIBS@

