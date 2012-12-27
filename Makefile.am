ACLOCAL_AMFLAGS=-I m4

AM_CXXFLAGS=@CL_CFLAGS@ @opencv_CFLAGS@ -std=c++0x -I${srcdir}/src \
			-I${builddir}/src @V4L2_CFLAGS@

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


TESTS=src/test/verifyDTCWT.sh src/test/verifyInterpolation.sh
TEST_EXTENSIONS=.sh


noinst_LTLIBRARIES = libdtcwt.la 
BUILT_SOURCES = src/KeypointDetector/findMaxKernel.h \
				src/KeypointDetector/concatKernel.h \
				src/KeypointDetector/EnergyMaps/interpMapKernel.h \
			src/KeypointDetector/EnergyMaps/interpPhaseMapKernel.h \
				src/KeypointDescriptor/extractDescriptorsKernel.h \
				src/MiscKernels/accumulateKernel.h 
libdtcwt_la_SOURCES = src/DTCWT/dtcwt.cc \
					  src/DTCWT/filterer.cc \
					  src/DTCWT/fourTree.cc \
					  src/KeypointDescriptor/extractDescriptors.cc \
					  src/KeypointDetector/findMax.cc \
					  src/KeypointDetector/concat.cc \
					  src/KeypointDetector/EnergyMaps/interpMap.cc \
				  src/KeypointDetector/EnergyMaps/interpPhaseMap.cc \
					  src/MiscKernels/accumulate.cc \
					  src/KeypointDetector/peakDetector.cc
libdtcwt_la_LIBADD = @CL_LIBS@ @opencv_LIBS@ libclutil.la 

noinst_LTLIBRARIES += libclutil.la
libclutil_la_SOURCES = src/util/clUtil.cc
libclutil_la_LIBADD = @CL_LIBS@ @opencv_LIBS@

noinst_PROGRAMS = testDTCWT
testDTCWT_SOURCES = src/test/test.cc 
testDTCWT_LDADD = libdtcwt.la @opencv_LIBS@

noinst_PROGRAMS += testDescriptor
testDescriptor_SOURCES = src/test/testDescriptor.cc 
testDescriptor_LDADD = libdtcwt.la @opencv_LIBS@

noinst_PROGRAMS += testFindMax
testFindMax_SOURCES = src/test/testFindMax.cc
testFindMax_LDADD = libdtcwt.la @opencv_LIBS@

noinst_PROGRAMS += testConcat
testConcat_SOURCES = src/test/testConcat.cc
testConcat_LDADD = libdtcwt.la @opencv_LIBS@

noinst_PROGRAMS += testAccumulate
testAccumulate_SOURCES = src/test/testAccumulate.cc 
testAccumulate_LDADD = libdtcwt.la @opencv_LIBS@

noinst_PROGRAMS += testPeakDetector
testPeakDetector_SOURCES = src/test/testPeakDetector.cc 
testPeakDetector_LDADD = libdtcwt.la @opencv_LIBS@

noinst_PROGRAMS += testPyramidSum
BUILT_SOURCES += src/MiscKernels/pyramidSumKernel.h 
testPyramidSum_SOURCES = src/MiscKernels/pyramidSum.cc \
						 src/test/testPyramidSum.cc 
testPyramidSum_LDADD = libdtcwt.la @opencv_LIBS@

#testOutput_SOURCES = src/test/testOutput.cc \
#					 src/MiscKernels/abs.cc
#testOutput_LDADD = libdtcwt.la libclutil.la -lsfml-window -lsfml-system \
#				   @opencv_LIBS@

noinst_PROGRAMS += displayDTCWT
BUILT_SOURCES += src/MiscKernels/absToRGBAKernel.h \
				 src/MiscKernels/greyscaleToRGBAKernel.h 
displayDTCWT_SOURCES = src/DisplayOutput/calculator.cc \
					   src/DisplayOutput/calculatorInterface.cc \
					   src/DisplayOutput/VBOBuffer.cc \
					   src/MiscKernels/abs.cc \
					   src/MiscKernels/greyscaleToRGBA.cc \
					   src/MiscKernels/absToRGBA.cc \
					   src/DisplayOutput/displayDTCWT.cc \
					   src/DisplayOutput/texture.cc \
					   src/DisplayOutput/viewer.cc \
					   src/DisplayOutput/VideoReader.cc \
					   src/DTCWT/energyMapEigen.cc
displayDTCWT_LDADD = libdtcwt.la libclutil.la -lsfml-window -lsfml-system \
				   @opencv_LIBS@ @V4L2_LIBS@ -lboost_thread


#testOutputEigenDetector_SOURCES = src/test/testOutputEigenDetector.cc \
					 #src/MiscKernels/abs.cc \
					 #src/DTCWT/energyMapEigen.cc
#testOutputEigenDetector_LDADD = libdtcwt.la libclutil.la -lsfml-window -lsfml-system \
				   #@opencv_LIBS@

noinst_PROGRAMS += testFilter
testFilter_SOURCES = src/test/testFilter.cc 
testFilter_LDADD = libdtcwt.la @opencv_LIBS@

noinst_PROGRAMS += testRescale
testRescale_SOURCES = src/test/testRescale.cc 
testRescale_LDADD = libdtcwt.la @opencv_LIBS@

noinst_PROGRAMS += verify
verify_SOURCES = src/test/verify.cc 
verify_LDADD = libdtcwt.la @opencv_LIBS@

noinst_PROGRAMS += verifyBasic
verifyBasic_SOURCES = src/test/verifyBasic.cc 
verifyBasic_LDADD = libdtcwt.la @opencv_LIBS@



noinst_LTLIBRARIES += libbufferfilter.la
BUILT_SOURCES += src/BufferTrials/FilterX/filterXKernel.h \
                 src/BufferTrials/FilterY/filterYKernel.h \
                 src/BufferTrials/DecimateFilterX/kernel.h \
                 src/BufferTrials/DecimateFilterY/kernel.h \
                 src/BufferTrials/PadX/kernel.h \
                 src/BufferTrials/PadY/kernel.h
libbufferfilter_la_SOURCES = \
		 src/BufferTrials/imageBuffer.cc \
		 src/BufferTrials/FilterX/filterX.cc \
		 src/BufferTrials/FilterY/filterY.cc \
		 src/BufferTrials/DecimateFilterX/decimateFilterX.cc \
		 src/BufferTrials/DecimateFilterY/decimateFilterY.cc \
		 src/BufferTrials/PadX/padX.cc \
		 src/BufferTrials/PadY/padY.cc 
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



TESTS += testFilterX testFilterY

check_PROGRAMS = testFilterX
testFilterX_SOURCES = src/BufferTrials/FilterX/testFilterX.cc 
testFilterX_LDADD   = libclutil.la libbufferfilter.la \
		     		  @CL_LIBS@ @opencv_LIBS@ @Eigen3_LIBS@
testFilterX_CXXFLAGS = @Eigen3_CFLAGS@ $(AM_CXXFLAGS)


check_PROGRAMS += testFilterY
testFilterY_SOURCES = src/BufferTrials/FilterY/testFilterY.cc 
testFilterY_LDADD   = libclutil.la libbufferfilter.la \
		     		  @CL_LIBS@ @opencv_LIBS@ @Eigen3_LIBS@
testFilterY_CXXFLAGS = @Eigen3_CFLAGS@ $(AM_CXXFLAGS)




TESTS += testDecimateFilterX testDecimateFilterY

check_PROGRAMS += testDecimateFilterX
testDecimateFilterX_SOURCES = \
	  src/BufferTrials/DecimateFilterX/testDecimateFilterX.cc 
testDecimateFilterX_LDADD   = libclutil.la libbufferfilter.la \
		     		  @CL_LIBS@ @opencv_LIBS@ @Eigen3_LIBS@
testDecimateFilterX_CXXFLAGS = @Eigen3_CFLAGS@ $(AM_CXXFLAGS)

check_PROGRAMS += testDecimateFilterY
testDecimateFilterY_SOURCES = \
	  src/BufferTrials/DecimateFilterY/testDecimateFilterY.cc 
testDecimateFilterY_LDADD   = libclutil.la libbufferfilter.la \
		     		  @CL_LIBS@ @opencv_LIBS@ @Eigen3_LIBS@
testDecimateFilterY_CXXFLAGS = @Eigen3_CFLAGS@ $(AM_CXXFLAGS)



