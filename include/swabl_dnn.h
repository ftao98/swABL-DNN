#ifndef _SWABL_DNN_H_
#define _SWABL_DNN_H_

#include <stdlib.h>
#include <stdio.h>

#if defined(__cplusplus)
extern "C" {
#endif

/* swABL-DNN return codes */
typedef enum {
    SWABL_DNN_STATUS_SUCCESS        = 0,
    SWABL_DNN_STATUS_FAIL           = 1,
} swabl_dnnStatus_t;

/* swABL-DNN data type */
typedef enum {
    SWABL_DNN_FLOAT     = 0,
    SWABL_DNN_DOUBLE    = 1,
} swabl_dnnDataType_t;

/* Tensor format */
typedef enum {
    SWABL_DNN_TENSOR_NCHW = 0,
    SWABL_DNN_TENSOR_NHWC = 1,
} swabl_dnnTensorFormat_t;

/* Convolution mode */
typedef enum {
    SWABL_DNN_CONVOLUTION = 0,
    SWABL_DNN_CROSS_CORRELATION = 1,
} swabl_dnnConvolutionMode_t;

/* Convolution Algorithm */
typedef enum {
    SWABL_DNN_CONVOLUTION_FWD_ALGO_DIRECT    = 0,
} swabl_dnnConvolutionFwdAlgo_t;

/* Pooling mode */
typedef enum {
    SWABL_DNN_MAX_POOLING_FWD_ALGO    = 0,
    SWABL_DNN_AVG_POOLING_FWD_ALGO    = 1,
} swabl_dnnPoolingMode_t;

/* Activation mode */
typedef enum {
    SWABL_DNN_ACTIVATION_SIGMOID    = 0,
    SWABL_DNN_ACTIVATION_RELU    = 1,
    SWABL_DNN_ACTIVATION_TANH =2,
    SWABL_DNN_ACTIVATION_CLIPPED_RELU =3,
    SWABL_DNN_ACTIVATION_ELU =4,
} swabl_dnnActivationMode_t;

/* Data structures to represent input/filter/output and the Neural Network Layer */
typedef struct swabl_dnnTensorStruct *swabl_dnnTensorDescriptor_t;
typedef struct swabl_dnnFilterStruct *swabl_dnnFilterDescriptor_t;
typedef struct swabl_dnnConvolutionStruct *swabl_dnnConvolutionDescriptor_t;

typedef struct swabl_dnnPoolingStruct *swabl_dnnPoolingDescriptor_t;
typedef struct swabl_dnnActivationStruct *swabl_dnnActivationDescriptor_t;

/* Create, set and destroy an instance of a generic Tensor */
swabl_dnnStatus_t
swabl_dnnCreateTensorDescriptor(swabl_dnnTensorDescriptor_t *ptensorDesc);

swabl_dnnStatus_t
swabl_dnnSetTensor4dDescriptor(
        swabl_dnnTensorDescriptor_t tensorDesc,
        swabl_dnnDataType_t dataType,
        swabl_dnnTensorFormat_t format,
        int n, int c, int h, int w
        );

swabl_dnnStatus_t
swabl_dnnDestroyTensorDescriptor(swabl_dnnTensorDescriptor_t tensorDesc);

/* Create, set and destroy an instance of Filter */
swabl_dnnStatus_t
swabl_dnnCreateFilterDescriptor(swabl_dnnFilterDescriptor_t *pfltDesc);

swabl_dnnStatus_t
swabl_dnnSetFilter4dDescriptor(
        swabl_dnnFilterDescriptor_t fltDesc,
        swabl_dnnDataType_t dataType,
        swabl_dnnTensorFormat_t format,
        int k, int c, int h, int w
        );

swabl_dnnStatus_t
swabl_dnnDestroyFilterDescriptor(swabl_dnnFilterDescriptor_t fltDesc);

/* Create, set and destroy an instance of convolution */
swabl_dnnStatus_t
swabl_dnnCreateConvolutionDescriptor(swabl_dnnConvolutionDescriptor_t *pconvDesc);

swabl_dnnStatus_t
swabl_dnnSetConvolution2dDescriptor(
        swabl_dnnConvolutionDescriptor_t convDesc,
        int pad_h, int pad_w, int stride_h, int stride_w,
        int dilation_h, int dilation_w, /* filter dilation */
        swabl_dnnConvolutionMode_t mode,
        swabl_dnnDataType_t computeType
        );

swabl_dnnStatus_t
swabl_dnnDestroyConvolutionDescriptor(swabl_dnnConvolutionDescriptor_t convDesc);

/* Convolution Layer - Forward and Backward */
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
        );



/* Create, set and destroy an instance of Pooling */
swabl_dnnStatus_t
swabl_dnnCreatePoolingDescriptor(swabl_dnnPoolingDescriptor_t *ppoolDesc);

swabl_dnnStatus_t
swabl_dnnSetPooling2dDescriptor(
        swabl_dnnPoolingDescriptor_t poolDesc,
        swabl_dnnPoolingMode_t mode,
        int windowHeight,
        int windowWidth,
        int pad_h,
        int pad_w,
        int stride_h,
        int stride_w
        );

swabl_dnnStatus_t
swabl_dnnDestroyPoolingDescriptor(swabl_dnnPoolingDescriptor_t poolDesc);

/* Pooling Layer - Forward and Backward */
swabl_dnnStatus_t
swabl_dnnPoolingForward(
        const void *alpha,
        const swabl_dnnTensorDescriptor_t inDesc,
        const void *in,
        const void *beta,
        const swabl_dnnTensorDescriptor_t outDesc,
        void *out,
        const swabl_dnnPoolingDescriptor_t poolDesc
        );



/* Create, set and destroy an instance of Activation */
swabl_dnnStatus_t
swabl_dnnCreateActivationDescriptor(swabl_dnnActivationDescriptor_t *pactDesc);

swabl_dnnStatus_t
swabl_dnnSetActivation2dDescriptor(
        swabl_dnnActivationDescriptor_t actDesc,
        swabl_dnnActivationMode_t mode,
        double coef
        );

swabl_dnnStatus_t
swabl_dnnDestroyActivationDescriptor(swabl_dnnActivationDescriptor_t actDesc);

/* Activation Layer - Forward and Backward */
swabl_dnnStatus_t
swabl_dnnActivationForward(
    swabl_dnnActivationDescriptor_t actDesc,
    const void *alpha,
    const swabl_dnnTensorDescriptor_t inDesc,
    const void *in,
    const void *beta,
    const swabl_dnnTensorDescriptor_t outDesc,
    void *out
        );


#if defined(__cplusplus)
}
#endif


#endif
