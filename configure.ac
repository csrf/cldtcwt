AC_PREREQ([2.69])

AC_INIT([test],[1.0],[timothy.gale@cantab.net])
AC_CONFIG_AUX_DIR([config])
AC_CONFIG_MACRO_DIR([m4])

dnl We want to make sure objects can be in appropriate directories to avoid 
AM_INIT_AUTOMAKE([1.9 foreign subdir-objects])

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


AC_CONFIG_FILES([Makefile])
AC_OUTPUT

