#ifndef _SWABL_DNNDESCRIPTOR_H_
#define _SWABL_DNNDESCRIPTOR_H_

#include "swabl_dnn.h"

/* Data structures to represent input/filter/output and the Neural Network Layer */
struct swabl_dnnTensorStruct {
    swabl_dnnDataType_t dataType_;
    swabl_dnnTensorFormat_t format_;
    int dim_[4];
};

struct swabl_dnnFilterStruct {
    swabl_dnnDataType_t dataType_;
    swabl_dnnTensorFormat_t format_;
    int dim_[4];
};

struct swabl_dnnConvolutionStruct{
    swabl_dnnConvolutionMode_t mode_;
    swabl_dnnDataType_t computeType_;
    int pad_[2];
    int stride_[2];
    int dilation_[2];
};

struct swabl_dnnPoolingStruct{
    swabl_dnnPoolingMode_t mode_;
    int window_[2];
    int pad_[2];
    int stride_[2];
};

struct swabl_dnnActivationStruct{
    swabl_dnnActivationMode_t mode_;
    double coef;
};


#endif
