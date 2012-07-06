ACLOCAL_AMFLAGS=-I m4

AM_CXXFLAGS=@CL_CFLAGS@ @opencv_CFLAGS@ -std=c++0x

TESTS=test/verifyDTCWT.sh
TEST_EXTENSIONS=.sh

bin_PROGRAMS=testDTCWT testFilter verify testOutput testFindMax testRescale \
			 verifyBasic

testDTCWT_SOURCES=test.cc filterer.cc clUtil.cc dtcwt.cc extractDescriptors.cc
testDTCWT_LDADD=@CL_LIBS@ @opencv_LIBS@

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

