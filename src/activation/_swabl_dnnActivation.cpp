#include "swabl_dnnActivation.h"
#include"math.h"

template <typename Type> Type Relu(Type in_x)
{
//    printf("in_x=%d\n",in_x);
    if (in_x<0) {
        return 0;
    }
    else
    {
        return in_x;
    }
}
template <typename Type> Type Sigmod(Type in_x)
{
    return (1/(1+exp(-1*in_x)));
}
template <typename Type> Type Tanh(Type in_x)
{
    Type e_x=exp(-1*in_x);
    Type ex=exp(in_x);
    return (Type)((ex-e_x)/(ex+e_x));
}


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
    )
{
    int i,j;

    int N, C, inH, inW, outH, outW;
    int n, c, inh, inw, outh, outw;
    int indx,outdx;

    N = inDesc->dim_[0];
    C = inDesc->dim_[1];
    inH = inDesc->dim_[2];
    inW = inDesc->dim_[3];

    outH = outDesc->dim_[2];
    outW = outDesc->dim_[3];

    int insd[]={C*inH*inW, inH*inW, inW, 1};
    int outsd[] = {C*outH*outW, outH*outW, outW, 1};
    printf("act_mode=%d\n",actDesc->mode_);
    switch(actDesc->mode_)
    {
        case 0:
            for(n=0;n<N;n++)
            {
                for(outh = 0; outh < outH; outh++)
                {
                    for(outw = 0; outw < outW; outw++)
                    {
                        for( c = 0; c < C; c++)
                        {
                            outdx = n*outsd[0] + c*outsd[1] + outh*outsd[2] + outw;
                            out[outdx] = Sigmod(in[outdx]);
                        }
                        
                    }
                }
            }
            break;
        case 1:
            for(n=0;n<N;n++)
            {
                for(outh = 0; outh < outH; outh++)
                {
                    for(outw = 0; outw < outW; outw++)
                    {
                        for( c = 0; c < C; c++)
                        {
                            outdx = n*outsd[0] + c*outsd[1] + outh*outsd[2] + outw;
                            out[outdx] = Relu(in[outdx]);
                        }
                        
                    }
                }
            }
            break;
        case 2:
            for(n=0;n<N;n++)
            {
                for(outh = 0; outh < outH; outh++)
                {
                    for(outw = 0; outw < outW; outw++)
                    {
                        for( c = 0; c < C; c++)
                        {
                            outdx = n*outsd[0] + c*outsd[1] + outh*outsd[2] + outw;
                            out[outdx] = Tanh(in[outdx]);
                        }
                        
                    }
                }
            }
            break;
        case 3:
            break;
        case 4:
            break;
        default:
            fprintf(stdout,"This is a error,mode_!\n");
            break;
    }
    return SWABL_DNN_STATUS_SUCCESS;
}
/* Instantiate template */

template swabl_dnnStatus_t
_swabl_dnnActivationForward(
    swabl_dnnActivationDescriptor_t actDesc,
    const float *alpha,
    const swabl_dnnTensorDescriptor_t inDesc,
    const float *in,
    const float *beta,
    const swabl_dnnTensorDescriptor_t outDesc,
    float *out
    );

template swabl_dnnStatus_t
_swabl_dnnActivationForward(
    swabl_dnnActivationDescriptor_t actDesc,
    const double *alpha,
    const swabl_dnnTensorDescriptor_t inDesc,
    const double *in,
    const double *beta,
    const swabl_dnnTensorDescriptor_t outDesc,
    double *out
    );

