#include <stdlib.h>
#include <stdio.h>
#include "swabl_dnn.h"

typedef float DataType;

int main(int argc, char *argv[])
{
    int i, j;

    // Pooling Configure
    int N, C, inH, inW;
    int outH, outW;

    N       = 1;//64;
    C       = 1;//16;
    inH     = 4;
    inW     = 4;
    outH = inH;
    outW = inW;

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
            fscanf(indata,"%f", (in+i*inW+j));
        }
    }
    fclose(indata);
    for(i=0;i<inH;i++)
    {
	for(j=0;j<inW;j++)
	{
	    printf("%f ",in[i*inW+j]);
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
    swabl_dnnActivationMode_t act_mode=SWABL_DNN_ACTIVATION_SIGMOID;
    printf("model=%d\n",act_mode);
    swabl_dnnActivationDescriptor_t act_desc;
	swabl_dnnCreateActivationDescriptor(&act_desc);
	swabl_dnnSetActivation2dDescriptor(act_desc,
			act_mode,0);

	swabl_dnnActivationForward(act_desc,&alpha,in_desc,in,&beta,out_desc,out);


    swabl_dnnDestroyTensorDescriptor(in_desc);
    swabl_dnnDestroyTensorDescriptor(out_desc);
	swabl_dnnDestroyActivationDescriptor(act_desc);
 
	for(i=0;i<outH;i++)
	{
		for(j=0;j<outW;j++)
		{
			printf("%f ",out[i*outW+j]);
		}
		printf("\n");
	}

    free(in);
    free(out);
	

    return 0;

}
