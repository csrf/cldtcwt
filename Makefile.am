ACLOCAL_AMFLAGS=-I m4

AM_CXXFLAGS=@CL_CFLAGS@ @opencv_CFLAGS@ -std=c++0x -I${srcdir}/src \
			-I${builddir}/src @V4L2_CFLAGS@

# For turning OpenCL source files into C++ source
include $(srcdir)/BuildRules/ClToCc.makerule

TESTS=src/test/verifyDTCWT.sh src/test/verifyInterpolation.sh
TEST_EXTENSIONS=.sh


noinst_LTLIBRARIES = libclutil.la
libclutil_la_SOURCES = src/util/clUtil.cc
libclutil_la_LDFLAGS = @CL_LIBS@ 


noinst_LTLIBRARIES += libclutilcv.la
libclutilcv_la_SOURCES = src/util/clUtilCV.cc
libclutilcv_la_LDFLAGS = @CL_LIBS@ @opencv_LIBS@
libclutilcv_la_CXXFLAGS = @opencv_CFLAGS@ $(AM_CXXFLAGS)


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
				src/KeypointDetector/EnergyMaps/Eigen/kernel.cc \
				src/KeypointDetector/EnergyMaps/EnergyMap/kernel.cc \
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



# Wrapper for the FFMPEG libraries
noinst_LTLIBRARIES += libavmm.la
libavmm_la_SOURCES = src/avmm/avmm.cc
libavmm_la_CXXFLAGS = $(FFMPEG_CFLAGS) -fPIC $(AM_CXXFLAGS)
libavmm_la_LIBADD = $(FFMPEG_LIBS)


# Test the speed of the FFMPEG libraries
noinst_PROGRAMS = speedTestFFMPEG
speedTestFFMPEG_SOURCES = src/SpeedComparisonFilters/ffmpeg/speedTest.cc
speedTestFFMPEG_LDADD = libavmm.la
speedTestFFMPEG_CXXFLAGS = $(FFMPEG_CFLAGS) $(AM_CXXFLAGS)



noinst_PROGRAMS += testDescriptor
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


# Display after input from VLC webcam
bin_PROGRAMS = displayDTCWT
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
displayDTCWT_LDFLAGS = @SFML_LIBS@ @V4L2_LIBS@ -lboost_thread 
displayDTCWT_CXXFLAGS = @SFML_CFLAGS@ $(AM_CXXFLAGS)


# Display after input from FFMPEG
bin_PROGRAMS += displayVideoDTCWT
displayVideoDTCWT_SOURCES = src/DisplayOutput/calculator.cc \
					   src/DisplayOutput/calculatorInterface.cc \
					   src/DisplayOutput/VBOBuffer.cc \
					   src/DisplayOutput/Abs/abs.cc \
					   src/DisplayOutput/GreyscaleToRGBA/greyscaleToRGBA.cc \
					   src/DisplayOutput/AbsToRGBA/absToRGBA.cc \
					   src/DisplayOutput/displayVideoDTCWT.cc \
					   src/DisplayOutput/texture.cc \
					   src/DisplayOutput/viewer.cc 
nodist_displayVideoDTCWT_SOURCES = \
	src/DisplayOutput/AbsToRGBA/kernel.cc \
	src/DisplayOutput/GreyscaleToRGBA/kernel.cc 
CLEANFILES += $(nodist_displayVideoDTCWT_SOURCES)
displayVideoDTCWT_LDADD = libdtcwt.la libclutil.la libkeypoints.la libavmm.la
displayVideoDTCWT_LDFLAGS = @SFML_LIBS@
displayVideoDTCWT_CXXFLAGS = @SFML_CFLAGS@ @FFMPEG_CFLAGS@ $(AM_CXXFLAGS)



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





# Speed tests for the non-very-used, but helpful for comparison with
# alternative techniques.

noinst_PROGRAMS += speedTestFilterXWithExtension
speedTestFilterXWithExtension_SOURCES = \
	src/SpeedComparisonFilters/FilterXWithSymExtension/speedTest.cc \
	src/SpeedComparisonFilters/FilterXWithSymExtension/filterX.cc 
speedTestFilterXWithExtension_LDADD = libclutil.la 
nodist_speedTestFilterXWithExtension_SOURCES = \
	src/SpeedComparisonFilters/FilterXWithSymExtension/kernel.cc

# Again, but this time allowing adjustment of the padding in the buffer
noinst_PROGRAMS += speedTestFilterXAdjustablePadding
speedTestFilterXAdjustablePadding_SOURCES = \
	src/SpeedComparisonFilters/FilterXAdjustablePadding/speedTest.cc \
	src/SpeedComparisonFilters/FilterXAdjustablePadding/filterX.cc 
speedTestFilterXAdjustablePadding_LDADD = libclutil.la 
nodist_speedTestFilterXAdjustablePadding_SOURCES = \
	src/SpeedComparisonFilters/FilterXAdjustablePadding/kernel.cc
CLEANFILES += $(nodist_speedTestFilterXAdjustablePadding_SOURCES)

# No-twiddle variant of DecimateFilterX, for comparison
check_PROGRAMS = testDecimateFilterX-Alt-RSLNT
TESTS += testDecimateFilterX-Alt-RSLNT
testDecimateFilterX_Alt_RSLNT_SOURCES = \
	  src/SpeedComparisonFilters/DecimateFilterX/ReversedSplitLoadNoTwiddle/decimateFilterX.cc \
	  src/SpeedComparisonFilters/DecimateFilterX/ReversedSplitLoadNoTwiddle/testDecimateFilterX.cc \
	  src/Filter/referenceImplementation.cc
nodist_testDecimateFilterX_Alt_RSLNT_SOURCES = \
	  src/SpeedComparisonFilters/DecimateFilterX/ReversedSplitLoadNoTwiddle/kernel.cc 
testDecimateFilterX_Alt_RSLNT_LDADD   = libclutil.la libbufferfilter.la
testDecimateFilterX_Alt_RSLNT_LDFLAGS = @Eigen3_LIBS@
testDecimateFilterX_Alt_RSLNT_CXXFLAGS = @Eigen3_CFLAGS@ $(AM_CXXFLAGS)
CLEANFILES += $(nodist_testDecimateFilterX_Alt_RSLNT_SOURCES)

bin_PROGRAMS += clDTCWT-SpeedTest-DecimateFilterX-Alt-RSLNT
clDTCWT_SpeedTest_DecimateFilterX_Alt_RSLNT_SOURCES = \
	src/SpeedComparisonFilters/DecimateFilterX/ReversedSplitLoadNoTwiddle/decimateFilterX.cc \
	src/SpeedComparisonFilters/DecimateFilterX/ReversedSplitLoadNoTwiddle/speedTestDecimateFilterX.cc 
clDTCWT_SpeedTest_DecimateFilterX_Alt_RSLNT_LDADD   = \
	libclutil.la libbufferfilter.la 
nodist_clDTCWT_SpeedTest_DecimateFilterX_Alt_RSLNT_SOURCES = \
	  src/SpeedComparisonFilters/DecimateFilterX/ReversedSplitLoadNoTwiddle/kernel.cc 
CLEANFILES += $(nodist_clDTCWT_SpeedTest_DecimateFilterX_Alt_RSLNT_SOURCES)

# Non-splitting and forward load variant of DecimateFilterX, for comparison
check_PROGRAMS += testDecimateFilterX-Alt-FL
TESTS += testDecimateFilterX-Alt-FL
testDecimateFilterX_Alt_FL_SOURCES = \
	  src/SpeedComparisonFilters/DecimateFilterX/ForwardLoad/decimateFilterX.cc \
	  src/SpeedComparisonFilters/DecimateFilterX/ForwardLoad/testDecimateFilterX.cc \
	  src/Filter/referenceImplementation.cc
nodist_testDecimateFilterX_Alt_FL_SOURCES = \
	  src/SpeedComparisonFilters/DecimateFilterX/ForwardLoad/kernel.cc 
testDecimateFilterX_Alt_FL_LDADD   = libclutil.la libbufferfilter.la
testDecimateFilterX_Alt_FL_LDFLAGS = @Eigen3_LIBS@
testDecimateFilterX_Alt_FL_CXXFLAGS = @Eigen3_CFLAGS@ $(AM_CXXFLAGS)
CLEANFILES += $(nodist_testDecimateFilterX_Alt_FL_SOURCES)

bin_PROGRAMS += clDTCWT-SpeedTest-DecimateFilterX-Alt-FL
clDTCWT_SpeedTest_DecimateFilterX_Alt_FL_SOURCES = \
	src/SpeedComparisonFilters/DecimateFilterX/ForwardLoad/decimateFilterX.cc \
	src/SpeedComparisonFilters/DecimateFilterX/ForwardLoad/speedTestDecimateFilterX.cc 
clDTCWT_SpeedTest_DecimateFilterX_Alt_FL_LDADD   = \
	libclutil.la libbufferfilter.la 
nodist_clDTCWT_SpeedTest_DecimateFilterX_Alt_FL_SOURCES = \
	  src/SpeedComparisonFilters/DecimateFilterX/ForwardLoad/kernel.cc 
CLEANFILES += $(nodist_clDTCWT_SpeedTest_DecimateFilterX_Alt_FL_SOURCES)



# Non-splitting and forward load variant of DecimateFilterX, for comparison
# This time, the filter coefficients are used in both directions
check_PROGRAMS += testDecimateFilterX-Alt-FLRF
TESTS += testDecimateFilterX-Alt-FLRF
testDecimateFilterX_Alt_FLRF_SOURCES = \
	  src/SpeedComparisonFilters/DecimateFilterX/ForwardLoadReverseFilter/decimateFilterX.cc \
	  src/SpeedComparisonFilters/DecimateFilterX/ForwardLoadReverseFilter/testDecimateFilterX.cc \
	  src/Filter/referenceImplementation.cc
nodist_testDecimateFilterX_Alt_FLRF_SOURCES = \
	  src/SpeedComparisonFilters/DecimateFilterX/ForwardLoadReverseFilter/kernel.cc 
testDecimateFilterX_Alt_FLRF_LDADD   = libclutil.la libbufferfilter.la
testDecimateFilterX_Alt_FLRF_LDFLAGS = @Eigen3_LIBS@
testDecimateFilterX_Alt_FLRF_CXXFLAGS = @Eigen3_CFLAGS@ $(AM_CXXFLAGS)
CLEANFILES += $(nodist_testDecimateFilterX_Alt_FLRF_SOURCES)

bin_PROGRAMS += clDTCWT-SpeedTest-DecimateFilterX-Alt-FLRF
clDTCWT_SpeedTest_DecimateFilterX_Alt_FLRF_SOURCES = \
	src/SpeedComparisonFilters/DecimateFilterX/ForwardLoadReverseFilter/decimateFilterX.cc \
	src/SpeedComparisonFilters/DecimateFilterX/ForwardLoadReverseFilter/speedTestDecimateFilterX.cc 
clDTCWT_SpeedTest_DecimateFilterX_Alt_FLRF_LDADD   = \
	libclutil.la libbufferfilter.la 
nodist_clDTCWT_SpeedTest_DecimateFilterX_Alt_FLRF_SOURCES = \
	  src/SpeedComparisonFilters/DecimateFilterX/ForwardLoadReverseFilter/kernel.cc 
CLEANFILES += $(nodist_clDTCWT_SpeedTest_DecimateFilterX_Alt_FLRF_SOURCES)





# Non-splitting but reversed load variant of DecimateFilterX, for comparison
check_PROGRAMS += testDecimateFilterX-Alt-RL
TESTS += testDecimateFilterX-Alt-RL
testDecimateFilterX_Alt_RL_SOURCES = \
	  src/SpeedComparisonFilters/DecimateFilterX/ReversedLoad/decimateFilterX.cc \
	  src/SpeedComparisonFilters/DecimateFilterX/ReversedLoad/testDecimateFilterX.cc \
	  src/Filter/referenceImplementation.cc
nodist_testDecimateFilterX_Alt_RL_SOURCES = \
	  src/SpeedComparisonFilters/DecimateFilterX/ReversedLoad/kernel.cc 
testDecimateFilterX_Alt_RL_LDADD   = libclutil.la libbufferfilter.la
testDecimateFilterX_Alt_RL_LDFLAGS = @Eigen3_LIBS@
testDecimateFilterX_Alt_RL_CXXFLAGS = @Eigen3_CFLAGS@ $(AM_CXXFLAGS)
CLEANFILES += $(nodist_testDecimateFilterX_Alt_RL_SOURCES)

bin_PROGRAMS += clDTCWT-SpeedTest-DecimateFilterX-Alt-RL
clDTCWT_SpeedTest_DecimateFilterX_Alt_RL_SOURCES = \
	src/SpeedComparisonFilters/DecimateFilterX/ReversedLoad/decimateFilterX.cc \
	src/SpeedComparisonFilters/DecimateFilterX/ReversedLoad/speedTestDecimateFilterX.cc 
clDTCWT_SpeedTest_DecimateFilterX_Alt_RL_LDADD   = \
	libclutil.la libbufferfilter.la 
nodist_clDTCWT_SpeedTest_DecimateFilterX_Alt_RL_SOURCES = \
	  src/SpeedComparisonFilters/DecimateFilterX/ReversedLoad/kernel.cc 
CLEANFILES += $(nodist_clDTCWT_SpeedTest_DecimateFilterX_Alt_RL_SOURCES)




# Straightforward copy, to work out how fast we can do the memory transfers
noinst_PROGRAMS += speedTestCopyImage
speedTestCopyImage_SOURCES = \
	src/SpeedComparisonFilters/CopyImage/speedTest.cc \
	src/SpeedComparisonFilters/CopyImage/copyImage.cc 
speedTestCopyImage_LDADD = libclutil.la 
nodist_speedTestCopyImage_SOURCES = \
	src/SpeedComparisonFilters/CopyImage/kernel.cc
CLEANFILES += $(nodist_speedTestCopyImage_SOURCES)

# Straightforward copy, to work out how fast we can do the memory transfers
# Here, rather than bothering with the image format, we just copy the 
# right amount of data
noinst_PROGRAMS += speedTestCopyImageLinearly
speedTestCopyImageLinearly_SOURCES = \
	src/SpeedComparisonFilters/CopyImageLinearly/speedTest.cc \
	src/SpeedComparisonFilters/CopyImageLinearly/copyImageLinearly.cc 
speedTestCopyImageLinearly_LDADD = libclutil.la 
nodist_speedTestCopyImageLinearly_SOURCES = \
	src/SpeedComparisonFilters/CopyImageLinearly/kernel.cc
CLEANFILES += $(nodist_speedTestCopyImageLinearly_SOURCES)




bin_PROGRAMS += clDTCWT-SpeedTest-BufferTrial
clDTCWT_SpeedTest_BufferTrial_SOURCES = src/Filter/speedTest.cc 
clDTCWT_SpeedTest_BufferTrial_LDADD   = libclutil.la libbufferfilter.la 

bin_PROGRAMS += clDTCWT-SpeedTest-DecimateFilterX
clDTCWT_SpeedTest_DecimateFilterX_SOURCES = \
	src/Filter/DecimateFilterX/speedTestDecimateFilterX.cc 
clDTCWT_SpeedTest_DecimateFilterX_LDADD   = \
	libclutil.la libbufferfilter.la 

bin_PROGRAMS += clDTCWT-SpeedTest-DecimateFilterY
clDTCWT_SpeedTest_DecimateFilterY_SOURCES = \
	src/Filter/DecimateFilterY/speedTestDecimateFilterY.cc 
clDTCWT_SpeedTest_DecimateFilterY_LDADD   = \
	libclutil.la libbufferfilter.la 

bin_PROGRAMS += clDTCWT-SpeedTest-DecimateTripleFilterX
clDTCWT_SpeedTest_DecimateTripleFilterX_SOURCES = \
	src/Filter/DecimateTripleFilterX/speedTest.cc 
clDTCWT_SpeedTest_DecimateTripleFilterX_LDADD   = \
	libclutil.la libbufferfilter.la 

bin_PROGRAMS += clDTCWT-SpeedTest-QuadToComplexDecimateFilterY
clDTCWT_SpeedTest_QuadToComplexDecimateFilterY_SOURCES = \
	src/Filter/QuadToComplexDecimateFilterY/speedTest.cc 
clDTCWT_SpeedTest_QuadToComplexDecimateFilterY_LDADD   = \
	libclutil.la libbufferfilter.la 

bin_PROGRAMS += clDTCWT-SpeedTest-DTCWT
clDTCWT_SpeedTest_DTCWT_SOURCES = src/test/test.cc 
clDTCWT_SpeedTest_DTCWT_LDADD   = libdtcwt.la libclutilcv.la




TESTS += testFilterX testFilterY

check_PROGRAMS += testFilterX
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


# MATLAB scripts and functions have their own folder to be installed into
matlabscriptdir = $(pkgdatadir)/MATLAB
dist_matlabscript_DATA = \
	MatlabScripts/getBufferTrialTimings.m \
	MatlabScripts/getDecimateFilterXTimings.m \
	MatlabScripts/runSpeedTestBufferTrial.m \
	MatlabScripts/runSpeedTestDecimateFilterX.m \
	MatlabScripts/runSpeedTestDecimateFilterY.m \
	MatlabScripts/runSpeedTestDecimateTripleFilterX.m \
	MatlabScripts/runSpeedTestQuadToComplexDecimateFilterY.m


