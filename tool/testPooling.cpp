#include <stdlib.h>
#include <stdio.h>
#include "swabl_dnn.h"

typedef int DataType;

int main(int argc, char *argv[])
{
    int i, j;

    // Pooling Configure
    int N, C, inH, inW;
    int R, S;//windowh,windoww
    int padh, padw, strideh, stridew;
    int outH, outW;

    N       = 1;//64;
    C       = 1;//16;
    inH     = 4;
    inW     = 4;
    R       = 2;
    S       = 2;
    padh    = 0;
    padw    = 0;
    strideh = 2;
    stridew = 2;
    outH = (inH+2*padh-R)/strideh + 1;
    outW = (inW+2*padw-S)/stridew + 1;

    DataType *in, *out;

    DataType alpha = 1.0f;
    DataType beta = 0.0f;

    int insize = N*C*inH*inW;
    int outsize = N*C*outH*outW;

    

    in = (DataType *)malloc(insize*sizeof(DataType));

    out = (DataType *)malloc(outsize*sizeof(DataType));
    //initial
    FILE *indata;
    indata=fopen("data4.txt","r");
    for(i = 0; i < inH; i++)
    {
        for(j = 0; j < inW; j++)
        {
            fscanf(indata,"%d", in+i*inW+j);
        }
    }
    fclose(indata);
    for(i=0;i<inH;i++)
    {
	for(j=0;j<inW;j++)
	{
	    printf("%d ",in[i*inW+j]);
	}
	printf("\n");
    }

    swabl_dnnTensorDescriptor_t in_desc;

    swabl_dnnTensorDescriptor_t out_desc;

    swabl_dnnDataType_t sw_data_type = SWABL_DNN_FLOAT;
    swabl_dnnTensorFormat_t sw_tensor_format = SWABL_DNN_TENSOR_NCHW;
    
    swabl_dnnCreateTensorDescriptor(&in_desc);
    swabl_dnnSetTensor4dDescriptor(in_desc, sw_data_type, sw_tensor_format,
            N, C, inH, inW);

    swabl_dnnCreateTensorDescriptor(&out_desc);
    swabl_dnnSetTensor4dDescriptor(out_desc, sw_data_type, sw_tensor_format,
            N, C, outH, outW);
    swabl_dnnPoolingMode_t pool_mode=SWABL_DNN_AVG_POOLING_FWD_ALGO;
    printf("model=%d\n",pool_mode);
    swabl_dnnPoolingDescriptor_t pool_desc;
	swabl_dnnCreatePoolingDescriptor(&pool_desc);
	swabl_dnnSetPooling2dDescriptor(pool_desc,
			pool_mode,R,S,padh,padw,strideh,stridew);

	swabl_dnnPoolingForward(&alpha,in_desc,in,&beta,out_desc,out,pool_desc);


    swabl_dnnDestroyTensorDescriptor(in_desc);
    swabl_dnnDestroyTensorDescriptor(out_desc);
	swabl_dnnDestroyPoolingDescriptor(pool_desc);
    
	for(i=0;i<outH;i++)
	{
		for(j=0;j<outW;j++)
		{
			printf("%d ",out[i*outW+j]);
		}
		printf("\n");
	}

    free(in);
    free(out);
	

    return 0;

}
