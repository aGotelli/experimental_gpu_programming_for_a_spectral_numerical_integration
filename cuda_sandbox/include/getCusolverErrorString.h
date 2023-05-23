#include <cusolverDn.h>

#ifndef GETCUSOLVERERRORSTRING_H
#define GETCUSOLVERERRORSTRING_H

const char* getCusolverErrorString(cusolverStatus_t error)
{
    switch (error)
    {
        case CUSOLVER_STATUS_SUCCESS:
            return "Success";
        case CUSOLVER_STATUS_NOT_INITIALIZED:
            return "Not initialized";
        case CUSOLVER_STATUS_ALLOC_FAILED:
            return "Allocation failed";
        case CUSOLVER_STATUS_INVALID_VALUE:
            return "Invalid value";
        case CUSOLVER_STATUS_ARCH_MISMATCH:
            return "Architecture mismatch";
        case CUSOLVER_STATUS_EXECUTION_FAILED:
            return "Execution failed";
        case CUSOLVER_STATUS_INTERNAL_ERROR:
            return "Internal error";
        case CUSOLVER_STATUS_MAPPING_ERROR:
            return "Mapping error";
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "Matrix type not supported";
        case CUSOLVER_STATUS_NOT_SUPPORTED:
            return "Operation not supported";
        default:
            return "Unknown error";
    }
}

#endif 