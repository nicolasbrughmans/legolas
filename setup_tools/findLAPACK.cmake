IF(LAPACK_LIBRARIES)
    SET(LAPACK_FOUND TRUE)
    RETURN()
ENDIF()

message(STATUS "Finding LAPACK")
find_package(LAPACK REQUIRED)