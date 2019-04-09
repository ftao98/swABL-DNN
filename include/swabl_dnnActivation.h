#ifndef _SWABL_DNNACTIVATION_H_
#define _SWABL_DNNACTIVATION_H_

#include "swabl_dnn.h"
#include "swabl_dnnDescriptor.h"

/* Forward algorithm */
template <typename Type>
swabl_dnnStatus_t
_swabl_dnnActivationForward(
    swabl_dnnActivationDescriptor_t actDesc,
    const Type *alpha,
    const swabl_dnnTensorDescriptor_t inDesc,
    const Type *in,
    const Type *beta,
    const swabl_dnnTensorDescriptor_t outDesc,
    Type *out
);
#endif
