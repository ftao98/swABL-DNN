#include "swabl_dnnConvolution.h"

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
        )
{
    int N, C, K, inH, inW, fltH, fltW, outH, outW;
    int padh, padw, strideh, stridew;
    int n, c, k, inh, inw, outh, outw, flth, fltw;
    int indx, fltdx, outdx;

    N = inDesc->dim_[0];
    C = inDesc->dim_[1];
    inH = inDesc->dim_[2];
    inW = inDesc->dim_[3];
    K = fltDesc->dim_[0];
    fltH = fltDesc->dim_[2];
    fltW = fltDesc->dim_[3];
    outH = outDesc->dim_[2];
    outW = outDesc->dim_[3];
    padh = convDesc->pad_[0];
    padw = convDesc->pad_[1];
    strideh = convDesc->stride_[0];
    stridew = convDesc->stride_[1];

    int insd[] = {C*inH*inW, inH*inW, inW, 1};
    int fltsd[] = {C*fltH*fltW, fltH*fltW, fltW, 1};
    int outsd[] = {K*outH*outW, outH*outW, outW, 1};
#if 1
    printf("[N C inH inW] = [%d %d %d %d]"
           "[K C fltH fltW = [%d %d %d %d]"
           "[N C outH outW] =[%d %d %d %d]"
           "[padh padw strideh stridew] = [%d %d %d %d]\n",
           N, C, inH, inW, K, C, fltH, fltW, N, C, outH, outW,
           padh, padw, strideh, stridew);
#endif
    for(n = 0; n < N; n++){
        for(k = 0; k < K; k++){
            for(outh = 0; outh < outH; outh++){
                for(outw = 0; outw < outW; outw++){
                    outdx = n*outsd[0] + k*outsd[1] + outh*outsd[2] + outw;
                    out[outdx] = 0;
                    for(c = 0; c < C; c++){
                        for(flth = 0; flth < fltH; flth++){
                            for(fltw = 0; fltw < fltW; fltw++){
                                inh = outh*strideh + flth - padh;
                                inw = outw*stridew + fltw - padw;
                                if((inh >= 0) && (inh < inH) && (inw >= 0) && (inw < inW)){
                                indx = n*insd[0] + c*insd[1] + (outh*strideh+flth-padh)*insd[2] + (outw*stridew+fltw-padw);
                                fltdx = k*fltsd[0] + c*fltsd[1] + flth*fltsd[2] + fltw;
                                out[outdx] += in[indx]*flt[fltdx];
                                }
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
swabl_dnnConvolutionForwardDirect(
        const float *alpha,
        const swabl_dnnTensorDescriptor_t inDesc,
        const float *in,
        const swabl_dnnFilterDescriptor_t fltDesc,
        const float *flt,
        const float *beta,
        const swabl_dnnTensorDescriptor_t outDesc,
        float *out,
        const swabl_dnnConvolutionDescriptor_t convDesc,
        swabl_dnnConvolutionFwdAlgo_t algo,
        void *workSpace
        );

template  swabl_dnnStatus_t
swabl_dnnConvolutionForwardDirect(
        const double *alpha,
        const swabl_dnnTensorDescriptor_t inDesc,
        const double *in,
        const swabl_dnnFilterDescriptor_t fltDesc,
        const double *flt,
        const double *beta,
        const swabl_dnnTensorDescriptor_t outDesc,
        double *out,
        const swabl_dnnConvolutionDescriptor_t convDesc,
        swabl_dnnConvolutionFwdAlgo_t algo,
        void *workSpace
        );
