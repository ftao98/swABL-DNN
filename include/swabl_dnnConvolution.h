#ifndef _SWABL_DNNCONVOLUTION_H_
#define _SWABL_DNNCONVOLUTION_H_

#include "swabl_dnn.h"
#include "swabl_dnnDescriptor.h"

/* Forward algorithm: direct */
template <typename Type>
swabl_dnnStatus_t
swabl_dnnConvolutionForwardDirect(
        const Type *alpha,
        const swabl_dnnTensorDescriptor_t inDesc,
        const Type *in,
        const swabl_dnnFilterDescriptor_t fltDesc,
        const Type *flt,
        const Type *beta,
        const swabl_dnnTensorDescriptor_t outDesc,
        Type *out,
        const swabl_dnnConvolutionDescriptor_t convDesc,
        swabl_dnnConvolutionFwdAlgo_t algo,
        void *workSpace
        );

#endif
