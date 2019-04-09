#include "swabl_dnnPooling.h"

#define intsize sizeof(int)

#define MAX(x,y) ((x) > (y) ? (x) : (y))
/* Convolution API */


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
        )
{
    int N, C, inH, inW, outH, outW;
    int padh, padw, strideh, stridew, windowh, windoww;
    int wh, ww;
    int n, c, inh, inw, outh, outw;
    int indx,outdx;

    N = inDesc->dim_[0];
    C = inDesc->dim_[1];
    inH = inDesc->dim_[2];
    inW = inDesc->dim_[3];

    outH = outDesc->dim_[2];
    outW = outDesc->dim_[3];

    padh = poolDesc->pad_[0];
    padw = poolDesc->pad_[1];
    strideh = poolDesc->stride_[0];
    stridew = poolDesc->stride_[1];
    windowh = poolDesc->window_[0];
    windoww = poolDesc->window_[1];

    int insd[]={C*inH*inW, inH*inW, inW, 1};
    int outsd[] = {C*outH*outW, outH*outW, outW, 1};

    for(n=0;n<N;n++)
    {
        for(outh = 0; outh < outH; outh++)
        {
            for(outw = 0; outw < outW; outw++)
            {
                for( c = 0; c < C; c++)
                {
                    outdx = n*outsd[0] + c*outsd[1] + outh*outsd[2] + outw;
                    out[outdx] = 0;
                    for( wh = 0; wh < windowh; wh++)
                    {
                        for( ww = 0; ww < windoww; ww++)
                        {
                            inh=outh*strideh+wh-padh;
                            inw=outw*stridew+ww-padw;
                            if((inh >= 0) && (inh < inH) && (inw >= 0) && (inw < inW))
                            {
                                indx=n*insd[0]+c*insd[1]+inh*inW+inw;
                                out[outdx]=MAX(out[outdx],in[indx]);
                            }
                        }
                    }  
                }
                
            }
        }
    }
    return SWABL_DNN_STATUS_SUCCESS;
}
/* Instantiate template */
template swabl_dnnStatus_t
swabl_dnnMaxPoolingForward(
        const float *alpha,
        const swabl_dnnTensorDescriptor_t inDesc,
        const float *in,
        const float *beta,
        const swabl_dnnTensorDescriptor_t outDesc,
        float *out,
        const swabl_dnnPoolingDescriptor_t poolDesc
        );

template swabl_dnnStatus_t
swabl_dnnMaxPoolingForward(
        const double *alpha,
        const swabl_dnnTensorDescriptor_t inDesc,
        const double *in,
        const double *beta,
        const swabl_dnnTensorDescriptor_t outDesc,
        double *out,
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
        )
{
    int N, C, inH, inW, outH, outW;
    int padh, padw, strideh, stridew, windowh, windoww;
    int wh, ww;
    int n, c, inh, inw, outh, outw;
    int indx,outdx;

    N = inDesc->dim_[0];
    C = inDesc->dim_[1];
    inH = inDesc->dim_[2];
    inW = inDesc->dim_[3];

    outH = outDesc->dim_[2];
    outW = outDesc->dim_[3];

    padh = poolDesc->pad_[0];
    padw = poolDesc->pad_[1];
    strideh = poolDesc->stride_[0];
    stridew = poolDesc->stride_[1];
    windowh = poolDesc->window_[0];
    windoww = poolDesc->window_[1];

    int insd[]={C*inH*inW, inH*inW, inW, 1};
    int outsd[] = {C*outH*outW, outH*outW, outW, 1};

    for(n=0;n<N;n++)
    {
        for(outh = 0; outh < outH; outh++)
        {
            for(outw = 0; outw < outW; outw++)
            {
                for( c = 0; c < C; c++)
                {
                    outdx = n*outsd[0] + c*outsd[1] + outh*outsd[2] + outw;
                    out[outdx] = 0;
                    for( wh = 0; wh < windowh; wh++)
                    {
                        for( ww = 0; ww < windoww; ww++)
                        {
                            inh=outh*strideh+wh-padh;
                            inw=outw*stridew+ww-padw;
                            if((inh >= 0) && (inh < inH) && (inw >= 0) && (inw < inW))
                            {
                                indx=n*insd[0]+c*insd[1]+inh*inW+inw;
                                out[outdx]+=in[indx];
                            }
                        }
                    } 
                    out[outdx]=(Type)out[outdx]/(windowh *windoww);
                }
                
            }
        }
    }
    return SWABL_DNN_STATUS_SUCCESS;
}
/* Instantiate template */
template swabl_dnnStatus_t
swabl_dnnAvgPoolingForward(
        const float *alpha,
        const swabl_dnnTensorDescriptor_t inDesc,
        const float *in,
        const float *beta,
        const swabl_dnnTensorDescriptor_t outDesc,
        float *out,
        const swabl_dnnPoolingDescriptor_t poolDesc
        );

template swabl_dnnStatus_t
swabl_dnnAvgPoolingForward(
        const double *alpha,
        const swabl_dnnTensorDescriptor_t inDesc,
        const double *in,
        const double *beta,
        const swabl_dnnTensorDescriptor_t outDesc,
        double *out,
        const swabl_dnnPoolingDescriptor_t poolDesc
        );