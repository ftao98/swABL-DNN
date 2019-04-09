#include "swabl_dnnConvolution.h"

/* Convolution API */
swabl_dnnStatus_t
swabl_dnnConvolutionForward(
        const void *alpha,
        const swabl_dnnTensorDescriptor_t inDesc,
        const void *in,
        const swabl_dnnFilterDescriptor_t fltDesc,
        const void *flt,
        const void *beta,
        const swabl_dnnTensorDescriptor_t outDesc,
        void *out,
        const swabl_dnnConvolutionDescriptor_t convDesc,
        swabl_dnnConvolutionFwdAlgo_t algo,
        void *workSpace
        )
{
    switch(algo){
        case SWABL_DNN_CONVOLUTION_FWD_ALGO_DIRECT:
            if(convDesc->computeType_ == SWABL_DNN_FLOAT){
                swabl_dnnConvolutionForwardDirect<float>(
                        (const float *)alpha, inDesc, (const float *)in,
                        fltDesc, (const float *)flt, (const float *)beta, outDesc, (float *)out,
                        convDesc, algo, workSpace);
            } else if(convDesc->computeType_ == SWABL_DNN_DOUBLE){
                swabl_dnnConvolutionForwardDirect<double>(
                        (const double*)alpha, inDesc, (const double*)in,
                        fltDesc, (const double*)flt, (const double*)beta, outDesc, (double*)out,
                        convDesc, algo, workSpace);
            } else {
                fprintf(stdout, "This is a error!\n");
            }
            break;

        default:
            fprintf(stdout, "This is a error!\n");
            break;
    }

    return SWABL_DNN_STATUS_SUCCESS;
}
