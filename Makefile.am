ACLOCAL_AMFLAGS=-I m4

AM_CXXFLAGS=@CL_CFLAGS@ @opencv_CFLAGS@ -std=c++0x

bin_PROGRAMS=test verify

test_SOURCES=test.cc filterer.cc clUtil.cc dtcwt.cc
test_LDADD=@CL_LIBS@ @opencv_LIBS@

verify_SOURCES=verify.cc filterer.cc clUtil.cc dtcwt.cc
verify_LDADD=@CL_LIBS@ @opencv_LIBS@
