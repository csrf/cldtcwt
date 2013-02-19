ACLOCAL_AMFLAGS=-I m4

AM_CXXFLAGS=@CL_CFLAGS@ @opencv_CFLAGS@ -std=c++0x -I${srcdir}/src \
			-I${builddir}/src @V4L2_CFLAGS@

# For turning OpenCL source files into C++ source
include $(srcdir)/BuildRules/ClToCc.makerule

TESTS=src/test/verifyDTCWT.sh src/test/verifyInterpolation.sh
TEST_EXTENSIONS=.sh


noinst_LTLIBRARIES = libdtcwt.la 
libdtcwt_la_SOURCES = src/DTCWT/dtcwt.cc \
					  src/DTCWT/filterer.cc \
					  src/DTCWT/fourTree.cc 
libdtcwt_la_LIBADD = @CL_LIBS@ @opencv_LIBS@ libclutil.la 



noinst_LTLIBRARIES += libkeypoints.la
libkeypoints_la_SOURCES = \
					  src/KeypointDescriptor/extractDescriptors.cc \
					  src/KeypointDetector/findMax.cc \
					  src/KeypointDetector/concat.cc \
					  src/KeypointDetector/EnergyMaps/InterpMap/interpMap.cc \
					  src/KeypointDetector/EnergyMaps/InterpPhaseMap/interpPhaseMap.cc \
					  src/KeypointDetector/EnergyMaps/Eigen/energyMapEigen.cc \
					  src/KeypointDetector/EnergyMaps/EnergyMap/energyMap.cc \
					  src/MiscKernels/accumulate.cc \
					  src/KeypointDetector/peakDetector.cc
nodist_libkeypoints_la_SOURCES = \
				src/KeypointDetector/concatKernel.cc \
				src/KeypointDetector/findMaxKernel.cc \
				src/KeypointDetector/EnergyMaps/InterpMap/kernel.cc \
				src/KeypointDetector/EnergyMaps/InterpPhaseMap/kernel.cc \
				src/KeypointDescriptor/kernel.cc \
				src/MiscKernels/accumulateKernel.cc 
CLEANFILES = $(noinst_libkeypoints_la_SOURCES)
libkeypoints_la_LIBADD = @CL_LIBS@ @opencv_LIBS@ libclutil.la 



noinst_LTLIBRARIES += libclutil.la
libclutil_la_SOURCES = src/util/clUtil.cc
libclutil_la_LIBADD = @CL_LIBS@ @opencv_LIBS@

noinst_PROGRAMS = testDescriptor
testDescriptor_SOURCES = src/test/testDescriptor.cc 
testDescriptor_LDADD = libdtcwt.la libkeypoints.la @opencv_LIBS@

noinst_PROGRAMS += testFindMax
testFindMax_SOURCES = src/test/testFindMax.cc
testFindMax_LDADD = libkeypoints.la @opencv_LIBS@

noinst_PROGRAMS += testConcat
testConcat_SOURCES = src/test/testConcat.cc
testConcat_LDADD = libkeypoints.la @opencv_LIBS@

noinst_PROGRAMS += testAccumulate
testAccumulate_SOURCES = src/test/testAccumulate.cc 
testAccumulate_LDADD = libkeypoints.la @opencv_LIBS@

noinst_PROGRAMS += testPeakDetector
testPeakDetector_SOURCES = src/test/testPeakDetector.cc 
testPeakDetector_LDADD = libkeypoints.la @opencv_LIBS@

noinst_PROGRAMS += testPyramidSum
testPyramidSum_SOURCES = src/MiscKernels/pyramidSum.cc \
						 src/test/testPyramidSum.cc 
nodist_testPyramidSum_SOURCES = \
			src/MiscKernels/pyramidSumKernel.cc 
CLEANFILES += $(nodist_testPyramidSum_SOURCES)
testPyramidSum_LDADD = libkeypoints.la @opencv_LIBS@

#testOutput_SOURCES = src/test/testOutput.cc \
#					 src/MiscKernels/abs.cc
#testOutput_LDADD = libdtcwt.la libclutil.la -lsfml-window -lsfml-system \
#				   @opencv_LIBS@

noinst_PROGRAMS += displayDTCWT
displayDTCWT_SOURCES = src/DisplayOutput/calculator.cc \
					   src/DisplayOutput/calculatorInterface.cc \
					   src/DisplayOutput/VBOBuffer.cc \
					   src/MiscKernels/abs.cc \
					   src/MiscKernels/greyscaleToRGBA.cc \
					   src/MiscKernels/absToRGBA.cc \
					   src/DisplayOutput/displayDTCWT.cc \
					   src/DisplayOutput/texture.cc \
					   src/DisplayOutput/viewer.cc \
					   src/DisplayOutput/VideoReader.cc 
nodist_displayDTCWT_SOURCES = \
	src/MiscKernels/absToRGBAKernel.cc \
	src/MiscKernels/greyscaleToRGBAKernel.cc 
CLEANFILES += $(nodist_displayDTCWT_SOURCES)
displayDTCWT_LDADD = libbufferfilter.la libclutil.la libkeypoints.la \
					 @SFML_LIBS@ @opencv_LIBS@ @V4L2_LIBS@ -lboost_thread 
displayDTCWT_CXXFLAGS = @SFML_CFLAGS@ $(AM_CXXFLAGS)


noinst_PROGRAMS += testFilter
testFilter_SOURCES = src/test/testFilter.cc 
testFilter_LDADD = libdtcwt.la @opencv_LIBS@

noinst_PROGRAMS += testRescale
testRescale_SOURCES = src/test/testRescale.cc 
testRescale_LDADD = libdtcwt.la @opencv_LIBS@

noinst_PROGRAMS += verify
verify_SOURCES = src/test/verify.cc 
verify_LDADD = libbufferfilter.la @opencv_LIBS@

noinst_PROGRAMS += verifyBasic
verifyBasic_SOURCES = src/test/verifyBasic.cc 
verifyBasic_LDADD = libbufferfilter.la @opencv_LIBS@



noinst_LTLIBRARIES += libbufferfilter.la
libbufferfilter_la_SOURCES = \
		 src/BufferTrials/imageBuffer.cc \
		 src/BufferTrials/FilterX/filterX.cc \
		 src/BufferTrials/FilterY/filterY.cc \
		 src/BufferTrials/DecimateFilterX/decimateFilterX.cc \
		 src/BufferTrials/DecimateFilterY/decimateFilterY.cc \
		 src/BufferTrials/QuadToComplexDecimateFilterY/q2cDecimateFilterY.cc \
		 src/BufferTrials/DecimateTripleFilterX/decimateTripleFilterX.cc \
		 src/BufferTrials/QuadToComplex/quadToComplex.cc \
		 src/BufferTrials/PadX/padX.cc \
		 src/BufferTrials/PadY/padY.cc \
		 src/BufferTrials/ImageToImageBuffer/imageToImageBuffer.cc \
		 src/BufferTrials/DTCWT/dtcwt.cc
nodist_libbufferfilter_la_SOURCES = \
		 src/BufferTrials/DecimateFilterX/kernel.cc \
		 src/BufferTrials/DecimateFilterY/kernel.cc \
		 src/BufferTrials/FilterX/kernel.cc \
		 src/BufferTrials/FilterY/kernel.cc \
		 src/BufferTrials/QuadToComplexDecimateFilterY/kernel.cc \
         src/BufferTrials/DecimateTripleFilterX/kernel.cc \
         src/BufferTrials/QuadToComplex/kernel.cc \
         src/BufferTrials/PadX/kernel.cc \
         src/BufferTrials/PadY/kernel.cc \
         src/BufferTrials/ImageToImageBuffer/kernel.cc
CLEANFILES += $(nodist_libbufferfilter_la_SOURCES)
libbufferfilter_la_LIBADD = @CL_LIBS@ libclutil.la 


noinst_PROGRAMS += speedTestBufferTrial
speedTestBufferTrial_SOURCES = src/BufferTrials/speedTest.cc 
speedTestBufferTrial_LDADD   = libclutil.la libbufferfilter.la \
							   @CL_LIBS@ @opencv_LIBS@ 

noinst_PROGRAMS += speedTestDecimateFilterX
speedTestDecimateFilterX_SOURCES = \
	   src/BufferTrials/DecimateFilterX/speedTestDecimateFilterX.cc 
speedTestDecimateFilterX_LDADD   = libclutil.la libbufferfilter.la \
							   @CL_LIBS@ @opencv_LIBS@ 

noinst_PROGRAMS += speedTestDecimateFilterY
speedTestDecimateFilterY_SOURCES = \
	   src/BufferTrials/DecimateFilterY/speedTestDecimateFilterY.cc 
speedTestDecimateFilterY_LDADD   = libclutil.la libbufferfilter.la \
							   @CL_LIBS@ @opencv_LIBS@ 

noinst_PROGRAMS += speedTestDecimateTripleFilterX
speedTestDecimateTripleFilterX_SOURCES = \
	   src/BufferTrials/DecimateTripleFilterX/speedTest.cc 
speedTestDecimateTripleFilterX_LDADD   = libclutil.la libbufferfilter.la \
							   @CL_LIBS@ @opencv_LIBS@ 

noinst_PROGRAMS += speedTestQuadToComplexDecimateFilterY
speedTestQuadToComplexDecimateFilterY_SOURCES = \
	   src/BufferTrials/QuadToComplexDecimateFilterY/speedTest.cc 
speedTestQuadToComplexDecimateFilterY_LDADD   = libclutil.la libbufferfilter.la \
							   @CL_LIBS@ @opencv_LIBS@ 

noinst_PROGRAMS += testDTCWT
testDTCWT_SOURCES = src/test/test.cc 
testDTCWT_LDADD = libbufferfilter.la libclutil.la @opencv_LIBS@




TESTS += testFilterX testFilterY

check_PROGRAMS = testFilterX
testFilterX_SOURCES = src/BufferTrials/FilterX/testFilterX.cc \
					  src/BufferTrials/referenceImplementation.cc
testFilterX_LDADD   = libclutil.la libbufferfilter.la \
		     		  @CL_LIBS@ @opencv_LIBS@ @Eigen3_LIBS@
testFilterX_CXXFLAGS = @Eigen3_CFLAGS@ $(AM_CXXFLAGS)


check_PROGRAMS += testFilterY
testFilterY_SOURCES = src/BufferTrials/FilterY/testFilterY.cc \
					  src/BufferTrials/referenceImplementation.cc
testFilterY_LDADD   = libclutil.la libbufferfilter.la \
		     		  @CL_LIBS@ @opencv_LIBS@ @Eigen3_LIBS@
testFilterY_CXXFLAGS = @Eigen3_CFLAGS@ $(AM_CXXFLAGS)




TESTS += testDecimateFilterX testDecimateFilterY

check_PROGRAMS += testDecimateFilterX
testDecimateFilterX_SOURCES = \
	  src/BufferTrials/DecimateFilterX/testDecimateFilterX.cc \
	  src/BufferTrials/referenceImplementation.cc
testDecimateFilterX_LDADD   = libclutil.la libbufferfilter.la \
		     		  @CL_LIBS@ @opencv_LIBS@ @Eigen3_LIBS@
testDecimateFilterX_CXXFLAGS = @Eigen3_CFLAGS@ $(AM_CXXFLAGS)

check_PROGRAMS += testDecimateFilterY
testDecimateFilterY_SOURCES = \
	  src/BufferTrials/DecimateFilterY/testDecimateFilterY.cc \
	  src/BufferTrials/referenceImplementation.cc
testDecimateFilterY_LDADD   = libclutil.la libbufferfilter.la \
		     		  @CL_LIBS@ @opencv_LIBS@ @Eigen3_LIBS@
testDecimateFilterY_CXXFLAGS = @Eigen3_CFLAGS@ $(AM_CXXFLAGS)


TESTS += testDecimateTripleFilterX 

check_PROGRAMS += testDecimateTripleFilterX
testDecimateTripleFilterX_SOURCES = \
	  src/BufferTrials/DecimateTripleFilterX/test.cc \
	  src/BufferTrials/referenceImplementation.cc
testDecimateTripleFilterX_LDADD   = libclutil.la libbufferfilter.la \
		     		  @CL_LIBS@ @opencv_LIBS@ @Eigen3_LIBS@
testDecimateTripleFilterX_CXXFLAGS = @Eigen3_CFLAGS@ $(AM_CXXFLAGS)


TESTS += testQuadToComplex testQuadToComplexDecimateFilterY

check_PROGRAMS += testQuadToComplex 
testQuadToComplex_SOURCES = \
	  src/BufferTrials/QuadToComplex/test.cc \
	  src/BufferTrials/referenceImplementation.cc
testQuadToComplex_LDADD   = libclutil.la libbufferfilter.la \
		     		  @CL_LIBS@ @opencv_LIBS@ @Eigen3_LIBS@
testQuadToComplex_CXXFLAGS = @Eigen3_CFLAGS@ $(AM_CXXFLAGS)

check_PROGRAMS += testQuadToComplexDecimateFilterY
testQuadToComplexDecimateFilterY_SOURCES = \
	  src/BufferTrials/QuadToComplexDecimateFilterY/test.cc \
	  src/BufferTrials/referenceImplementation.cc
testQuadToComplexDecimateFilterY_LDADD   = libclutil.la libbufferfilter.la \
		     		  @CL_LIBS@ @opencv_LIBS@ @Eigen3_LIBS@
testQuadToComplexDecimateFilterY_CXXFLAGS = @Eigen3_CFLAGS@ $(AM_CXXFLAGS)


