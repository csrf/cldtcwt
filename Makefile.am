ACLOCAL_AMFLAGS=-I m4

AM_CXXFLAGS=@CL_CFLAGS@ @opencv_CFLAGS@ -std=c++0x

bin_PROGRAMS=test
test_SOURCES=test.cc filterer.cc clUtil.cc dtcwt.cc
test_LDADD=@CL_LIBS@ @opencv_LIBS@

