#ifndef _SWABL_DNNPOOLING_H_
#define _SWABL_DNNPOOLING_H_

#include "swabl_dnn.h"
#include "swabl_dnnDescriptor.h"

/* Forward algorithm */
template <typename Type>
swabl_dnnStatus_t
swabl_dnnMaxPoolingForward(
        const Type *alpha,
        const swabl_dnnTensorDescriptor_t inDesc,
        const Type *in,
        const Type *beta,
        const swabl_dnnTensorDescriptor_t outDesc,
        Type *out,
        const swabl_dnnPoolingDescriptor_t poolDesc
        );

template <typename Type>
swabl_dnnStatus_t
swabl_dnnAvgPoolingForward(
        const Type *alpha,
        const swabl_dnnTensorDescriptor_t inDesc,
        const Type *in,
        const Type *beta,
        const swabl_dnnTensorDescriptor_t outDesc,
        Type *out,
        const swabl_dnnPoolingDescriptor_t poolDesc
        );
#endif
