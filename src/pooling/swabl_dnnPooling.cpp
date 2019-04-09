#include "swabl_dnnPooling.h"

/* Pooling API*/
swabl_dnnStatus_t
swabl_dnnPoolingForward(
        const void *alpha,
        const swabl_dnnTensorDescriptor_t inDesc,
        const void *in,
        const void *beta,
        const swabl_dnnTensorDescriptor_t outDesc,
        void *out,
        const swabl_dnnPoolingDescriptor_t poolDesc
        )
{
//	printf("mode=%d\n",poolDesc->mode_);
    switch(poolDesc->mode_)
    {
        case 0:
            if(inDesc->dataType_==SWABL_DNN_FLOAT)
            {
                swabl_dnnMaxPoolingForward<float>(
                    (const float *)alpha,inDesc,(const float *)in,
                    (const float *)beta,outDesc,(float *)out,
                    poolDesc
                );
            }
            else if (inDesc->dataType_==SWABL_DNN_DOUBLE) 
            {
                swabl_dnnMaxPoolingForward<double>(
                    (const double *)alpha,inDesc,(const double *)in,
                    (const double *)beta,outDesc,(double *)out,
                    poolDesc
                );
            }
            else 
            {
                fprintf(stdout, "This is a error1!\n");
            }
	    break;
        case 1:
            if(inDesc->dataType_==SWABL_DNN_FLOAT)
            {
                swabl_dnnAvgPoolingForward<float>(
                    (const float *)alpha,inDesc,(const float *)in,
                    (const float *)beta,outDesc,(float *)out,
                    poolDesc
                );
            }
            else if (inDesc->dataType_==SWABL_DNN_DOUBLE) 
            {
                swabl_dnnAvgPoolingForward<double>(
                    (const double *)alpha,inDesc,(const double *)in,
                    (const double *)beta,outDesc,(double *)out,
                    poolDesc
                );
            }
            else 
            {
                fprintf(stdout, "This is a error2!\n");
            }
	    break;
        default:
            fprintf(stdout,"This is a error3!\n");
            break;
    }
    return SWABL_DNN_STATUS_SUCCESS;
}
