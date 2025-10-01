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
                output[k][y][x]=bias[k];//bias
                for(int j=0;j<CONV1_DIM;j++){//kernel y axis
                    for(int i=0;i<CONV1_DIM;i++){//kernel x axis
                        output[k][y][x]+=input[0][y+j][x+i]*kernel[k][0][j][i];
                    }
                }
                if(output[k][y][x]<0) output[k][y][x]=0; //ReLU
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
void Pool1_24x24x20_2x2x20_2_0(	float 	input[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH], 	    // IN
    float 	output[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH]);		// OUT
{
    int filtre[2][2];
    for(int i=0;i<POOL1_NBOUTPUT;i+=1){//depth of pool1
        for(int y=0;y<POOL1_HEIGHT;y++){//pool1_outpu y axis
            for(int x=0;x<POOL1_WIDTH;x++){//pool1_outpu x axis
                for(int k=0;k<CONV1_HEIGHT-2;k+=2){//conv1_output matrice
                    filtre[0][0]=input[i][k][k];
                    filtre[0][1]=input[i][k][k+1];
                    filtre[1][0]=input[i][k+1][k];
                    filtre[1][1]=input[i][k+1][k+1];
                    output[y][x]=max(filtre);
                }
            }
        }
    }
}
/*
parametres
input:20x12x12
kernel:40x20x5x5
bias:40
output:40x8x8
*/
void Conv2_12x12x20_5x5x40_1_0(	float input[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH], 	            // IN
    float kernel[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM], 	// IN
    float bias[CONV2_NBOUTPUT], 						                    // IN
    float output[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH]); 		        // OUT
{    
    for(int k=0;k<CONV2_NBOUTPUT;k++){//depth of output
        for(int m=0;m<POOL1_NBOUTPUT;m++){ 
        for(int y=0;y<CONV2_HEIGHT;y++){//output y axis
            for(int x=0;x<CONV2_WIDTH;x++){//output x axis
                output[k][y][x]=bias[k];//bias
                for(int j=0;j<CONV2_DIM;j++){//kernel y axis
                    for(int i=0;i<CONV2_DIM;i++){//kernel x axis
                        output[k][y][x]+=input[m][y+j][x+i]*kernel[k][m][j][i];
                    }
                }
            }
        }
    }
    if(output[k][y][x]<0) output[k][y][x]=0; //ReLU
    }
}
/*
parametres:
input:40x8x8
output:40x4x4

*/
void Pool2_8x8x40_2x2x40_2_0(	float 	input[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH], 	    // IN
    float 	output[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH]);		// OUT
{
    int filtre[2][2];
    for(int i=0;i<POOL2_NBOUTPUT;i+=1){//depth of pool2
        for(int y=0;y<POOL2_HEIGHT;y++){//pool1_outpu y axis
            for(int x=0;x<POOL2_WIDTH;x++){//pool1_outpu x axis
                for(int k=0;k<CONV2_HEIGHT-2;k+=2){//conv1_output matrice
                    filtre[0][0]=input[i][k][k];
                    filtre[0][1]=input[i][k][k+1];
                    filtre[1][0]=input[i][k+1][k];
                    filtre[1][1]=input[i][k+1][k+1];
                    output[y][x]=max(filtre);
                }
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
    for(int k=0;k<FC1_NBOUTPUT;k++){
        output[k]=bias[k];
        for(int i=0;i<POOL2_NBOUTPUT;i++){
            for(int y=0;y<POOL2_HEIGH;y++){
                for(int x=0;x<POOL2_WIDTH;x++){
                    output[k]+=kernel[k][i][y][x]*input[i][y][x];
                }

            }
        }
        if(output[k]<0) output[k]=0;
    }

}
/*
parametres
input:400
kernel:10x400
bias:10
output:10
*/
void Fc2_400_10(	float 	input[FC1_NBOUTPUT], 			        // IN
    float 	kernel[FC2_NBOUTPUT][FC1_NBOUTPUT],	    // IN
    float 	bias[FC2_NBOUTPUT],			            // IN
    float 	output[FC2_NBOUTPUT]); 			        // OUT
{
    for(int k=0;k<FC2_NBOUTPUT;k++){
        output[k]=bias[k];
        for(int i=0;i<FC1_NBOUTPUT;i++){
            output[k]+=input[i]*kernel[k][i];
        }
        if(output[k]<0) output[k]=0;
}
