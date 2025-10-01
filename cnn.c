#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "lenet_cnn_float.h"
/*
parametres
input:28x28x1
conv1_kernel:5x5x1x20
conv1_bias:20
conv1_output:24x24x20
*/
Conv1_28x28x1_5x5x20_1_0(input, conv1_kernel, conv1_bias, conv1_output){
    for(int k=0;k<CONV1_NBOUTPUT;k++){//depth of output 
        for(int y=0;y<CONV1_HEIGHT;y++){//output y axis
            for(int x=0;x<CONV1_WIDTH;x++){//output x axis
                conv1_output[k][y][x]=conv1_bias[k];//bias
                for(int j=0;j<CONV1_DIM;j++){//kernel y axis
                    for(int i=0;i<CONV1_DIM;i++){//kernel x axis
                        conv1_output[k][y][x]+=input[0][y+j][x+i]*conv1_kernel[k][0][j][i];
                    }
                }
                if(conv1_output[k][y][x]<0) conv1_output[k][y][x]=0; //ReLU
            }
        }
    }
}

/*
parametres
conv1_output:24x24x20
pool1_output:12x12x20
filtre:2x2
decalage:2
*/
int max(int filtre[2][2]){
    int max1,max2;
    max1=(filtre[0][0]>filtre[0][1])?filtre[0][0]:filtre[0][1];
    max2=(filtre[1][0]>filtre[1][1])?filtre[1][0]:filtre[1][1];
    return (max1>max2)?max1:max2;
}
Pool1_24x24x20_2x2x20_2_0(conv1_output, pool1_output){
    int filtre[2][2];
    for(int i=0;i<POOL1_NBOUTPUT;i+=1){//depth of pool1
        for(int y=0;y<POOL1_HEIGHT;y++){//pool1_outpu y axis
            for(int x=0;x<POOL1_WIDTH;x++){//pool1_outpu x axis
                for(int k=0;k<22;k+=2){//conv1_output matrice
                    filtre[0][0]=conv1_output[i][k][k];
                    filtre[0][1]=conv1_output[i][k][k+1];
                    filtre[1][0]=conv1_output[i][k+1][k];
                    filtre[1][1]=conv1_output[i][k+1][k+1];
                    pool1_output[y][x]=max(filtre);
                }
            }
        }
    }
}
/*
parametres
pool1_output:12x12x20
conv2_kernel:5x5x40
conv2_bias:40
conv2_output:8x8x40

*/
Conv2_12x12x20_5x5x40_1_0(pool1_output, conv2_kernel, conv2_bias, conv2_output){
    for(int k=0;k<CONV2_NBOUTPUT;k++){//depth of output 
        for(int y=0;y<CONV2_HEIGHT;y++){//output y axis
            for(int x=0;x<CONV2_WIDTH;x++){//output x axis
                conv2_output[k][y][x]=conv2_bias[k];//bias
                for(int j=0;j<CONV2_DIM;j++){//kernel y axis
                    for(int i=0;i<CONV2_DIM;i++){//kernel x axis
                        conv2_output[k][y][x]+=pool1_output[0][y+j][x+i]*conv1_kernel[k][0][j][i];
                    }
                }
                if(conv1_output[k][y][x]<0) conv1_output[k][y][x]=0; //ReLU
            }
        }
    }
}
/*
parametres
input:40x4x4
kernel:400x40x4x4
bias:400
output:400
*/

void Fc1_40_400(	float 	input[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH], 			        // IN
    float 	kernel[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],	// IN
    float 	bias[FC1_NBOUTPUT],							                        // IN
    float 	output[FC1_NBOUTPUT]); 							                    // OUT
{

}