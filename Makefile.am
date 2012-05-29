ACLOCAL_AMFLAGS=-I m4

AM_CXXFLAGS=@CL_CFLAGS@ @opencv_CFLAGS@ -std=c++0x

bin_PROGRAMS=test testFilter verify testOutput testFindMax testRescale

test_SOURCES=test.cc filterer.cc clUtil.cc dtcwt.cc
test_LDADD=@CL_LIBS@ @opencv_LIBS@

testFindMax_SOURCES=testFindMax.cc filterer.cc clUtil.cc dtcwt.cc findMax.cc
testFindMax_LDADD=@CL_LIBS@ @opencv_LIBS@

testOutput_SOURCES=testOutput.cc filterer.cc clUtil.cc dtcwt.cc
testOutput_LDADD=@CL_LIBS@ @opencv_LIBS@ -lglut

testFilter_SOURCES=testFilter.cc filterer.cc clUtil.cc dtcwt.cc
testFilter_LDADD=@CL_LIBS@ @opencv_LIBS@

testRescale_SOURCES=testRescale.cc filterer.cc clUtil.cc dtcwt.cc
testRescale_LDADD=@CL_LIBS@ @opencv_LIBS@

verify_SOURCES=verify.cc filterer.cc clUtil.cc dtcwt.cc
verify_LDADD=@CL_LIBS@ @opencv_LIBS@
