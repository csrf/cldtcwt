ACLOCAL_AMFLAGS=-I m4

AM_CXXFLAGS=@CL_CFLAGS@ @opencv_CFLAGS@ -std=c++0x -I${srcdir}/src \
			-I${builddir}/src @V4L2_CFLAGS@

# For turning OpenCL source files into C++ source
include $(srcdir)/BuildRules/ClToCc.makerule

TESTS=src/test/verifyDTCWT.sh src/test/verifyInterpolation.sh
TEST_EXTENSIONS=.sh


noinst_LTLIBRARIES = libclutil.la
libclutil_la_SOURCES = src/util/clUtil.cc
libclutil_la_LDFLAGS = @CL_LIBS@ @opencv_LIBS@


noinst_LTLIBRARIES += libbufferfilter.la
libbufferfilter_la_SOURCES = \
		 src/Filter/imageBuffer.cc \
		 src/Filter/FilterX/filterX.cc \
		 src/Filter/FilterY/filterY.cc \
		 src/Filter/DecimateFilterX/decimateFilterX.cc \
		 src/Filter/DecimateFilterY/decimateFilterY.cc \
		 src/Filter/QuadToComplexDecimateFilterY/q2cDecimateFilterY.cc \
		 src/Filter/DecimateTripleFilterX/decimateTripleFilterX.cc \
		 src/Filter/QuadToComplex/quadToComplex.cc \
		 src/Filter/PadX/padX.cc \
		 src/Filter/PadY/padY.cc \
		 src/Filter/ImageToImageBuffer/imageToImageBuffer.cc 
nodist_libbufferfilter_la_SOURCES = \
		 src/Filter/DecimateFilterX/kernel.cc \
		 src/Filter/DecimateFilterY/kernel.cc \
		 src/Filter/FilterX/kernel.cc \
		 src/Filter/FilterY/kernel.cc \
		 src/Filter/QuadToComplexDecimateFilterY/kernel.cc \
         src/Filter/DecimateTripleFilterX/kernel.cc \
         src/Filter/QuadToComplex/kernel.cc \
         src/Filter/PadX/kernel.cc \
         src/Filter/PadY/kernel.cc \
         src/Filter/ImageToImageBuffer/kernel.cc
CLEANFILES = $(nodist_libbufferfilter_la_SOURCES)
libbufferfilter_la_LIBADD = libclutil.la 
libbufferfilter_la_LDFLAGS = @CL_LIBS@



noinst_LTLIBRARIES += libdtcwt.la 
libdtcwt_la_SOURCES = src/DTCWT/dtcwt.cc 
libdtcwt_la_LIBADD = libclutil.la libbufferfilter.la
libdtcwt_la_LDFLAGS = @CL_LIBS@ 



noinst_LTLIBRARIES += libkeypoints.la
libkeypoints_la_SOURCES = \
					  src/KeypointDetector/FindMax/findMax.cc \
					  src/KeypointDetector/Concat/concat.cc \
					  src/KeypointDetector/EnergyMaps/InterpMap/interpMap.cc \
					  src/KeypointDetector/EnergyMaps/InterpPhaseMap/interpPhaseMap.cc \
					  src/KeypointDetector/EnergyMaps/Eigen/energyMapEigen.cc \
					  src/KeypointDetector/EnergyMaps/EnergyMap/energyMap.cc \
					  src/KeypointDetector/EnergyMaps/PyramidSum/pyramidSum.cc \
					  src/KeypointDetector/Accumulate/accumulate.cc \
					  src/KeypointDetector/peakDetector.cc \
					  src/KeypointDescriptor/extractDescriptors.cc 
nodist_libkeypoints_la_SOURCES = \
				src/KeypointDetector/Concat/kernel.cc \
				src/KeypointDetector/FindMax/kernel.cc \
				src/KeypointDetector/EnergyMaps/InterpMap/kernel.cc \
				src/KeypointDetector/EnergyMaps/InterpPhaseMap/kernel.cc \
				src/KeypointDetector/EnergyMaps/PyramidSum/kernel.cc \
				src/KeypointDetector/Accumulate/kernel.cc \
				src/KeypointDescriptor/kernel.cc 
CLEANFILES += $(nodist_libkeypoints_la_SOURCES)
libkeypoints_la_LIBADD = libclutil.la 


noinst_LTLIBRARIES += librescale.la
librescale_la_SOURCES = \
             	src/MiscKernels/Rescale/rescale.cc 
librescale_la_LIBADD = libclutil.la 


noinst_PROGRAMS = testDescriptor
testDescriptor_SOURCES = src/test/testDescriptor.cc 
testDescriptor_LDADD = libdtcwt.la libkeypoints.la
testDescriptor_LDFLAGS = @opencv_LIBS@
testDescriptor_CXXFLAGS = @opencv_CFLAGS@ $(AM_CXXFLAGS)

noinst_PROGRAMS += testFindMax
testFindMax_SOURCES = src/test/testFindMax.cc
testFindMax_LDADD = libkeypoints.la 

noinst_PROGRAMS += testConcat
testConcat_SOURCES = src/test/testConcat.cc
testConcat_LDADD = libkeypoints.la

noinst_PROGRAMS += testAccumulate
testAccumulate_SOURCES = src/test/testAccumulate.cc 
testAccumulate_LDADD = libkeypoints.la

noinst_PROGRAMS += testPeakDetector
testPeakDetector_SOURCES = src/test/testPeakDetector.cc 
testPeakDetector_LDADD = libkeypoints.la

noinst_PROGRAMS += testPyramidSum
testPyramidSum_SOURCES = src/test/testPyramidSum.cc 
testPyramidSum_LDADD = libkeypoints.la

noinst_PROGRAMS += displayDTCWT
displayDTCWT_SOURCES = src/DisplayOutput/calculator.cc \
					   src/DisplayOutput/calculatorInterface.cc \
					   src/DisplayOutput/VBOBuffer.cc \
					   src/DisplayOutput/Abs/abs.cc \
					   src/DisplayOutput/GreyscaleToRGBA/greyscaleToRGBA.cc \
					   src/DisplayOutput/AbsToRGBA/absToRGBA.cc \
					   src/DisplayOutput/displayDTCWT.cc \
					   src/DisplayOutput/texture.cc \
					   src/DisplayOutput/viewer.cc \
					   src/DisplayOutput/VideoReader.cc 
nodist_displayDTCWT_SOURCES = \
	src/DisplayOutput/AbsToRGBA/kernel.cc \
	src/DisplayOutput/GreyscaleToRGBA/kernel.cc 
CLEANFILES += $(nodist_displayDTCWT_SOURCES)
displayDTCWT_LDADD = libdtcwt.la libclutil.la libkeypoints.la 
displayDTCWT_LDFLAGS = @SFML_LIBS@ @opencv_LIBS@ @V4L2_LIBS@ -lboost_thread 
displayDTCWT_CXXFLAGS = @SFML_CFLAGS@ $(AM_CXXFLAGS)


noinst_PROGRAMS += testRescale
testRescale_SOURCES = src/test/testRescale.cc 
testRescale_LDADD = librescale.la libclutilcv.la
testRescale_LDFLAGS = @opencv_LIBS@
testRescale_CXXFLAGS = @opencv_CFLAGS@ $(AM_CXXFLAGS)

noinst_PROGRAMS += verify
verify_SOURCES = src/test/verify.cc 
verify_LDADD = libdtcwt.la
verify_LDFLAGS = @opencv_LIBS@
verify_CXXFLAGS = @opencv_CFLAGS@ $(AM_CXXFLAGS)

noinst_PROGRAMS += verifyBasic
verifyBasic_SOURCES = src/test/verifyBasic.cc 
verifyBasic_LDADD = libdtcwt.la
verifyBasic_LDFLAGS = @opencv_LIBS@
verifyBasic_CXXFLAGS = @opencv_CFLAGS@ $(AM_CXXFLAGS)





noinst_PROGRAMS += speedTestBufferTrial
speedTestBufferTrial_SOURCES = src/Filter/speedTest.cc 
speedTestBufferTrial_LDADD   = libclutil.la libbufferfilter.la 

noinst_PROGRAMS += speedTestDecimateFilterX
speedTestDecimateFilterX_SOURCES = \
	   src/Filter/DecimateFilterX/speedTestDecimateFilterX.cc 
speedTestDecimateFilterX_LDADD   = libclutil.la libbufferfilter.la 

noinst_PROGRAMS += speedTestDecimateFilterY
speedTestDecimateFilterY_SOURCES = \
	   src/Filter/DecimateFilterY/speedTestDecimateFilterY.cc 
speedTestDecimateFilterY_LDADD   = libclutil.la libbufferfilter.la 

noinst_PROGRAMS += speedTestDecimateTripleFilterX
speedTestDecimateTripleFilterX_SOURCES = \
	   src/Filter/DecimateTripleFilterX/speedTest.cc 
speedTestDecimateTripleFilterX_LDADD   = libclutil.la libbufferfilter.la 

noinst_PROGRAMS += speedTestQuadToComplexDecimateFilterY
speedTestQuadToComplexDecimateFilterY_SOURCES = \
	   src/Filter/QuadToComplexDecimateFilterY/speedTest.cc 
speedTestQuadToComplexDecimateFilterY_LDADD   = libclutil.la libbufferfilter.la 

noinst_PROGRAMS += testDTCWT
testDTCWT_SOURCES = src/test/test.cc 
testDTCWT_LDADD   = libdtcwt.la libclutilcv.la
testDTCWT_LDFLAGS = @opencv_LIBS@
testDTCWT_CXXFLAGS = @opencv_CFLAGS@ $(AM_CXXFLAGS)




TESTS += testFilterX testFilterY

check_PROGRAMS = testFilterX
testFilterX_SOURCES = src/Filter/FilterX/testFilterX.cc \
					  src/Filter/referenceImplementation.cc
testFilterX_LDADD   = libclutil.la libbufferfilter.la 
testFilterX_LDFLAGS = @Eigen3_LIBS@
testFilterX_CXXFLAGS = @Eigen3_CFLAGS@ $(AM_CXXFLAGS)


check_PROGRAMS += testFilterY
testFilterY_SOURCES = src/Filter/FilterY/testFilterY.cc \
					  src/Filter/referenceImplementation.cc
testFilterY_LDADD   = libclutil.la libbufferfilter.la 
testFilterY_LDFLAGS = @Eigen3_LIBS@
testFilterY_CXXFLAGS = @Eigen3_CFLAGS@ $(AM_CXXFLAGS)




TESTS += testDecimateFilterX testDecimateFilterY

check_PROGRAMS += testDecimateFilterX
testDecimateFilterX_SOURCES = \
	  src/Filter/DecimateFilterX/testDecimateFilterX.cc \
	  src/Filter/referenceImplementation.cc
testDecimateFilterX_LDADD   = libclutil.la libbufferfilter.la 
testDecimateFilterX_LDFLAGS = @Eigen3_LIBS@
testDecimateFilterX_CXXFLAGS = @Eigen3_CFLAGS@ $(AM_CXXFLAGS)

check_PROGRAMS += testDecimateFilterY
testDecimateFilterY_SOURCES = \
	  src/Filter/DecimateFilterY/testDecimateFilterY.cc \
	  src/Filter/referenceImplementation.cc
testDecimateFilterY_LDADD   = libclutil.la libbufferfilter.la 
testDecimateFilterY_LDFLAGS = @Eigen3_LIBS@
testDecimateFilterY_CXXFLAGS = @Eigen3_CFLAGS@ $(AM_CXXFLAGS)


TESTS += testDecimateTripleFilterX 

check_PROGRAMS += testDecimateTripleFilterX
testDecimateTripleFilterX_SOURCES = \
	  src/Filter/DecimateTripleFilterX/test.cc \
	  src/Filter/referenceImplementation.cc
testDecimateTripleFilterX_LDADD   = libclutil.la libbufferfilter.la 
testDecimateTripleFilterX_LDFLAGS = @Eigen3_LIBS@
testDecimateTripleFilterX_CXXFLAGS = @Eigen3_CFLAGS@ $(AM_CXXFLAGS)


TESTS += testQuadToComplex testQuadToComplexDecimateFilterY

check_PROGRAMS += testQuadToComplex 
testQuadToComplex_SOURCES = \
	  src/Filter/QuadToComplex/test.cc \
	  src/Filter/referenceImplementation.cc
testQuadToComplex_LDADD   = libclutil.la libbufferfilter.la 
testQuadToComplex_LDFLAGS = @Eigen3_LIBS@
testQuadToComplex_CXXFLAGS = @Eigen3_CFLAGS@ $(AM_CXXFLAGS)

check_PROGRAMS += testQuadToComplexDecimateFilterY
testQuadToComplexDecimateFilterY_SOURCES = \
	  src/Filter/QuadToComplexDecimateFilterY/test.cc \
	  src/Filter/referenceImplementation.cc
testQuadToComplexDecimateFilterY_LDADD   = libclutil.la libbufferfilter.la
testQuadToComplexDecimateFilterY_LDFLAGS = @Eigen3_LIBS@
testQuadToComplexDecimateFilterY_CXXFLAGS = @Eigen3_CFLAGS@ $(AM_CXXFLAGS)


