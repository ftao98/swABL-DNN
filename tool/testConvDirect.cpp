#include <stdlib.h>
#include <stdio.h>
#include "swabl_dnn.h"

typedef float DataType;

int main(int argc, char *argv[])
{
    int i, j, k;

    // Convolution Configure
    int N, C, K, inH, inW;
    int R, S;
    int padh, padw, strideh, stridew;
    int outH, outW;

    N       = 1;//64;
    C       = 1;//16;
    K       = 1;//64;
    inH     = 10;
    inW     = 10;
    R       = 3;
    S       = 3;
    padh    = 1;
    padw    = 1;
    strideh = 1;
    stridew = 1;
    outH = (inH+2*padh-R)/strideh + 1;
    outW = (inW+2*padw-S)/stridew + 1;  

    DataType *in, *flt, *out;
    int dilationh = 1;
    int dilationw = 1;
    DataType alpha = 1.0f;
    DataType beta = 0.0f;

    int insize = N*C*inH*inW;
    int fltsize = K*C*R*S;
    int outsize = N*K*outH*outW;
    
    in = (DataType *)malloc(insize*sizeof(DataType));
    flt = (DataType *)malloc(fltsize*sizeof(DataType));
    out = (DataType *)malloc(outsize*sizeof(DataType));

    for(k = 0; k < insize; k++)
        in[k] = 0.1*(rand()%4);
    for(k = 0; k < fltsize; k++)
        flt[k] = 0.1*(rand()%4);

    swabl_dnnTensorDescriptor_t in_desc;
    swabl_dnnFilterDescriptor_t flt_desc;
    swabl_dnnTensorDescriptor_t out_desc;

    swabl_dnnDataType_t sw_data_type = SWABL_DNN_FLOAT;
    swabl_dnnTensorFormat_t sw_tensor_format = SWABL_DNN_TENSOR_NCHW;
    
    swabl_dnnCreateTensorDescriptor(&in_desc);
    swabl_dnnSetTensor4dDescriptor(in_desc, sw_data_type, sw_tensor_format,
            N, C, inH, inW);
    swabl_dnnCreateFilterDescriptor(&flt_desc);
    swabl_dnnSetFilter4dDescriptor(flt_desc, sw_data_type, sw_tensor_format,
            K, C, R, S);
    swabl_dnnCreateTensorDescriptor(&out_desc);
    swabl_dnnSetTensor4dDescriptor(out_desc, sw_data_type, sw_tensor_format,
            N, K, outH, outW);

    swabl_dnnConvolutionMode_t conv_mode = SWABL_DNN_CONVOLUTION;
    swabl_dnnConvolutionDescriptor_t conv_desc;
    swabl_dnnCreateConvolutionDescriptor(&conv_desc);
    swabl_dnnSetConvolution2dDescriptor(conv_desc,
            padh, padw, strideh, stridew, dilationh, dilationw,
            conv_mode, sw_data_type
            );

    swabl_dnnConvolutionForward(
            &alpha, in_desc, in, flt_desc, flt, &beta, out_desc, out,
            conv_desc, SWABL_DNN_CONVOLUTION_FWD_ALGO_DIRECT, NULL
            );

    swabl_dnnDestroyTensorDescriptor(in_desc);
    swabl_dnnDestroyFilterDescriptor(flt_desc);
    swabl_dnnDestroyTensorDescriptor(out_desc);
    swabl_dnnDestroyConvolutionDescriptor(conv_desc);

    free(in);
    free(flt);
    free(out);

    return 0;

}
