AC_PREREQ([2.69])

AC_INIT([clDTCWT],[0.1],[timothy.gale@cantab.net])
AC_CONFIG_AUX_DIR([config])
AC_CONFIG_MACRO_DIR([m4])

dnl We want to make sure objects can be in appropriate directories to avoid 
AM_INIT_AUTOMAKE([1.11 foreign subdir-objects silent-rules parallel-tests])

LT_INIT

AC_PROG_CXX

AX_CHECK_GL
AX_CHECK_GLUT

AX_CHECK_CL
if test "X$no_cl" = "Xyes"; then
    AC_MSG_FAILURE([You need OpenCL])
fi

PKG_CHECK_MODULES([opencv], [opencv glew])
PKG_CHECK_MODULES([V4L2], [libv4l2])
PKG_CHECK_MODULES([Eigen3], [eigen3])
PKG_CHECK_MODULES([SFML], [sfml-system >= 2.0 sfml-window >= 2.0])

# Make sure we can use FFMPEG for video input
PKG_CHECK_MODULES([FFMPEG], [libavcodec >= 54
                             libavformat >= 54
                             libavutil >= 52
                             libswscale >= 2.1.100])

AC_CONFIG_FILES([Makefile])
AC_OUTPUT

