#include "swabl_dnnDescriptor.h"

/* Create, set and destroy an generic Tensor */
swabl_dnnStatus_t
swabl_dnnCreateTensorDescriptor(swabl_dnnTensorDescriptor_t *ptensorDesc)
{
    *ptensorDesc = (struct swabl_dnnTensorStruct *)malloc(sizeof(struct swabl_dnnTensorStruct));
    
    return SWABL_DNN_STATUS_SUCCESS;
}

swabl_dnnStatus_t
swabl_dnnSetTensor4dDescriptor(
        swabl_dnnTensorDescriptor_t tensorDesc,
        swabl_dnnDataType_t dataType,
        swabl_dnnTensorFormat_t format,
        int n, int c, int h, int w
        )
{
    tensorDesc->dataType_ = dataType;
    tensorDesc->format_ = format;
    tensorDesc->dim_[0] = n;
    tensorDesc->dim_[1] = c;
    tensorDesc->dim_[2] = h;
    tensorDesc->dim_[3] = w;
    
    return SWABL_DNN_STATUS_SUCCESS;
}

swabl_dnnStatus_t
swabl_dnnDestroyTensorDescriptor(swabl_dnnTensorDescriptor_t tensorDesc)
{
    if(tensorDesc != NULL)
        free(tensorDesc);

    return SWABL_DNN_STATUS_SUCCESS;
}

/* Create, set and destroy instance of Filter */
swabl_dnnStatus_t
swabl_dnnCreateFilterDescriptor(swabl_dnnFilterDescriptor_t *pfltDesc)
{
    *pfltDesc = (struct swabl_dnnFilterStruct*)malloc(sizeof(struct swabl_dnnFilterStruct));

    return SWABL_DNN_STATUS_SUCCESS;
}

swabl_dnnStatus_t
swabl_dnnSetFilter4dDescriptor(
        swabl_dnnFilterDescriptor_t fltDesc,
        swabl_dnnDataType_t dataType,
        swabl_dnnTensorFormat_t format,
        int k, int c, int h, int w
        )
{
    fltDesc->dataType_ = dataType;
    fltDesc->format_ = format;
    fltDesc->dim_[0] = k;
    fltDesc->dim_[1] = c;
    fltDesc->dim_[2] = h;
    fltDesc->dim_[3] = w;

    return SWABL_DNN_STATUS_SUCCESS;
}

swabl_dnnStatus_t
swabl_dnnDestroyFilterDescriptor(swabl_dnnFilterDescriptor_t fltDesc)
{
    if(fltDesc != NULL)
        free(fltDesc);

    return SWABL_DNN_STATUS_SUCCESS;
}

/* Create, set and destroy an instance of convolution */
swabl_dnnStatus_t
swabl_dnnCreateConvolutionDescriptor(swabl_dnnConvolutionDescriptor_t *pconvDesc)
{
    *pconvDesc = (struct swabl_dnnConvolutionStruct*)malloc(sizeof(struct swabl_dnnConvolutionStruct));

    return SWABL_DNN_STATUS_SUCCESS;
}

swabl_dnnStatus_t
swabl_dnnSetConvolution2dDescriptor(
        swabl_dnnConvolutionDescriptor_t convDesc,
        int pad_h, int pad_w, int stride_h, int stride_w,
        int dilation_h, int dilation_w,
        swabl_dnnConvolutionMode_t mode,
        swabl_dnnDataType_t computeType
        )
{
    convDesc->mode_ = mode;
    convDesc->computeType_ = computeType;
    convDesc->pad_[0] = pad_h;
    convDesc->pad_[1] = pad_w;
    convDesc->stride_[0] = stride_h;
    convDesc->stride_[1] = stride_w;
    convDesc->dilation_[0] = dilation_h;
    convDesc->dilation_[1] = dilation_w;

    return SWABL_DNN_STATUS_SUCCESS;
}

swabl_dnnStatus_t
swabl_dnnDestroyConvolutionDescriptor(swabl_dnnConvolutionDescriptor_t convDesc)
{
    if(convDesc != NULL)
        free(convDesc);

    return SWABL_DNN_STATUS_SUCCESS;
}


/* Create, set and destroy an instance of pooling */
swabl_dnnStatus_t
swabl_dnnCreatePoolingDescriptor(swabl_dnnPoolingDescriptor_t *ppoolDesc)
{
    *ppoolDesc = (struct swabl_dnnPoolingStruct*)malloc(sizeof(struct swabl_dnnPoolingStruct));

    return SWABL_DNN_STATUS_SUCCESS;
}

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
        )
{
    poolDesc->mode_ = mode;
    poolDesc->pad_[0] = pad_h;
    poolDesc->pad_[1] = pad_w;
    poolDesc->stride_[0] = stride_h;
    poolDesc->stride_[1] = stride_w;
    poolDesc->window_[0]=windowHeight;
    poolDesc->window_[1]=windowWidth;


    return SWABL_DNN_STATUS_SUCCESS;
}

swabl_dnnStatus_t
swabl_dnnDestroyPoolingDescriptor(swabl_dnnPoolingDescriptor_t poolDesc)
{
    if(poolDesc != NULL)
        free(poolDesc);

    return SWABL_DNN_STATUS_SUCCESS;
}

/* Create, set and destroy an instance of activation */
swabl_dnnStatus_t
swabl_dnnCreateActivationDescriptor(swabl_dnnActivationDescriptor_t *pactDesc)
{
    *pactDesc = (struct swabl_dnnActivationStruct*)malloc(sizeof(struct swabl_dnnActivationStruct));

    return SWABL_DNN_STATUS_SUCCESS;
}

swabl_dnnStatus_t
swabl_dnnSetActivation2dDescriptor(
        swabl_dnnActivationDescriptor_t actDesc,
        swabl_dnnActivationMode_t mode,
        double coef
        )
{
    actDesc->mode_ = mode;
    actDesc->coef  =coef;

    return SWABL_DNN_STATUS_SUCCESS;
}

swabl_dnnStatus_t
swabl_dnnDestroyActivationDescriptor(swabl_dnnActivationDescriptor_t actDesc)
{
    if(actDesc != NULL)
        free(actDesc);

    return SWABL_DNN_STATUS_SUCCESS;
}