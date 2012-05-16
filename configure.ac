AC_PREREQ([2.59])

AC_INIT([test], [1.0], [timothy.gale@cantab.net])
AC_CONFIG_AUX_DIR([config])
AC_CONFIG_MACRO_DIR([m4])

AM_INIT_AUTOMAKE([1.9 foreign])

AC_PROG_CXX

AX_CHECK_GL
AX_CHECK_GLUT

AX_CHECK_CL
if test "X$no_cl" = "Xyes"; then
    AC_MSG_FAILURE([You need OpenCL])
fi

PKG_CHECK_MODULES([opencv], [opencv glew])


AC_CONFIG_FILES([Makefile])
AC_OUTPUT

