# Checks OpenCV is present, and works (using C++)
#
AC_DEFUN([AX_CHECK_OPENCV],
[
    AC_REQUIRE([AC_PROG_CC])
    AC_REQUIRE([AC_PROG_CPP])

    ax_save_CPPFLAGS="$CPPFLAGS"
    ax_save_LIBS="$LIBS"

    # Use C++ for these tests
    AC_LANG_PUSH([C++])


    CXXFLAGS="${OPENCV_CXXFLAGS} ${CXX

    AC_CHECK_HEADERS([cv.h])



    AC_LANG_POP([C++])

    CPPFLAGS=${ax_save_CPPFLAGS}

    OPENCV_CFLAGS=`pkg-config --cflags opencv`
    OPENCV_LIBS=`pkg-config --libs opencv`

    AC_SUBST([OPENCV_CFLAGS])
    AC_SUBST([OPENCV_LIBS])    
])
