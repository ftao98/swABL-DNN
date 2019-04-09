#include "swabl_dnnActivation.h"

/* Activation API*/
swabl_dnnStatus_t
swabl_dnnActivationForward(
    swabl_dnnActivationDescriptor_t actDesc,
    const void *alpha,
    const swabl_dnnTensorDescriptor_t inDesc,
    const void *in,
    const void *beta,
    const swabl_dnnTensorDescriptor_t outDesc,
    void *out
        )
{
    if(inDesc->dataType_==SWABL_DNN_FLOAT)
    {
        _swabl_dnnActivationForward<float>(actDesc,
            (const float *)alpha,inDesc,(const float *)in,
            (const float *)beta,outDesc,(float *)out
        );
    }
    else if (inDesc->dataType_==SWABL_DNN_DOUBLE)
    {
        _swabl_dnnActivationForward<double>(actDesc,
            (const double *)alpha,inDesc,(const double *)in,
            (const double *)beta,outDesc,(double *)out
        );
    }
    else 
    {
        fprintf(stdout, "This is a error,type!\n");
    }
    return SWABL_DNN_STATUS_SUCCESS;
}
