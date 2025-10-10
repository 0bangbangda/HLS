
/**
  ******************************************************************************
  * @file    main_fixed.c
  * @brief   LeNet CNN avec arithmétique fixed-point pour MNIST
  * @note    Conversion du code float vers fixed-point (Q4.12)
  ******************************************************************************
  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "hdf5.h"
#include "lenet_cnn_fixed.h"

// Variables globales
unsigned char 	REF_IMG[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH];
fixed_t 		INPUT_NORM[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH];
fixed_t 		CONV1_KERNEL[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM];
fixed_t 		CONV1_BIAS[CONV1_NBOUTPUT];
fixed_t 		CONV2_KERNEL[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM];
fixed_t 		CONV2_BIAS[CONV2_NBOUTPUT];
fixed_t 		FC1_KERNEL[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH];
fixed_t 		FC1_BIAS[FC1_NBOUTPUT];
fixed_t 		FC2_KERNEL[FC2_NBOUTPUT][FC1_NBOUTPUT];
fixed_t 		FC2_BIAS[FC2_NBOUTPUT];
fixed_t 		FC2_OUTPUT[FC2_NBOUTPUT];
fixed_t			SOFTMAX_OUTPUT[FC2_NBOUTPUT];

/**
  * @brief  Lecture fichier PGM
  */
void ReadPgmFile(char *filename, unsigned char *data) {
    FILE *pgm_file;
    char line[256];
    int width, height, max_gray;
    int i, j;
    
    pgm_file = fopen(filename, "rb");
    if (!pgm_file) {
        printf("Erreur: Impossible d'ouvrir %s\n", filename);
        exit(1);
    }
    
    // Lire l'en-tête PGM
    fgets(line, sizeof(line), pgm_file); // P5 ou P2
    fgets(line, sizeof(line), pgm_file); // Commentaire ou dimensions
    while (line[0] == '#') {
        fgets(line, sizeof(line), pgm_file);
    }
    sscanf(line, "%d %d", &width, &height);
    fgets(line, sizeof(line), pgm_file); // Max gray value
    sscanf(line, "%d", &max_gray);
    
    // Lire les données
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            data[i * width + j] = fgetc(pgm_file);
        }
    }
    
    fclose(pgm_file);
}

/**
  * @brief  Lecture des poids Conv1 depuis HDF5 avec conversion fixed-point
  */
void ReadConv1Weights_fixed(char *filename, char *dataset_name, fixed_t weights[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM])
{
    hid_t file_id, dataset_id;
    herr_t status;
    float *buffer;
    int i, j, k, l;
    int size = CONV1_NBOUTPUT * IMG_DEPTH * CONV1_DIM * CONV1_DIM;
    
    buffer = (float *)malloc(size * sizeof(float));
    
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset_id = H5Dopen(file_id, dataset_name, H5P_DEFAULT);
    status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
    
    // Conversion float -> fixed-point
    int idx = 0;
    for(k = 0; k < CONV1_NBOUTPUT; k++) {
        for(l = 0; l < IMG_DEPTH; l++) {
            for(i = 0; i < CONV1_DIM; i++) {
                for(j = 0; j < CONV1_DIM; j++) {
                    weights[k][l][i][j] = FLOAT_TO_FIXED(buffer[idx++]);
                }
            }
        }
    }
    
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    free(buffer);
}

/**
  * @brief  Lecture des biais Conv1 avec conversion fixed-point
  */
void ReadConv1Bias_fixed(char *filename, char *dataset_name, fixed_t bias[CONV1_NBOUTPUT])
{
    hid_t file_id, dataset_id;
    herr_t status;
    float buffer[CONV1_NBOUTPUT];
    int i;
    
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset_id = H5Dopen(file_id, dataset_name, H5P_DEFAULT);
    status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
    
    for(i = 0; i < CONV1_NBOUTPUT; i++) {
        bias[i] = FLOAT_TO_FIXED(buffer[i]);
    }
    
    H5Dclose(dataset_id);
    H5Fclose(file_id);
}

/**
  * @brief  Lecture des poids Conv2 avec conversion fixed-point
  */
void ReadConv2Weights_fixed(char *filename, char *dataset_name, 
    fixed_t weights[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM])
{
    hid_t file_id, dataset_id;
    herr_t status;
    float *buffer;
    int i, j, k, l;
    int size = CONV2_NBOUTPUT * POOL1_NBOUTPUT * CONV2_DIM * CONV2_DIM;
    
    buffer = (float *)malloc(size * sizeof(float));
    
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset_id = H5Dopen(file_id, dataset_name, H5P_DEFAULT);
    status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
    
    int idx = 0;
    for(k = 0; k < CONV2_NBOUTPUT; k++) {
        for(l = 0; l < POOL1_NBOUTPUT; l++) {
            for(i = 0; i < CONV2_DIM; i++) {
                for(j = 0; j < CONV2_DIM; j++) {
                    weights[k][l][i][j] = FLOAT_TO_FIXED(buffer[idx++]);
                }
            }
        }
    }
    
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    free(buffer);
}

/**
  * @brief  Lecture des biais Conv2 avec conversion fixed-point
  */
void ReadConv2Bias_fixed(char *filename, char *dataset_name, fixed_t bias[CONV2_NBOUTPUT])
{
    hid_t file_id, dataset_id;
    herr_t status;
    float buffer[CONV2_NBOUTPUT];
    int i;
    
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset_id = H5Dopen(file_id, dataset_name, H5P_DEFAULT);
    status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
    
    for(i = 0; i < CONV2_NBOUTPUT; i++) {
        bias[i] = FLOAT_TO_FIXED(buffer[i]);
    }
    
    H5Dclose(dataset_id);
    H5Fclose(file_id);
}

/**
  * @brief  Lecture des poids FC1 avec conversion fixed-point
  */
void ReadFc1Weights_fixed(char *filename, char *dataset_name,
    fixed_t weights[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH])
{
    hid_t file_id, dataset_id;
    herr_t status;
    float *buffer;
    int i, j, k, l;
    int size = FC1_NBOUTPUT * POOL2_NBOUTPUT * POOL2_HEIGHT * POOL2_WIDTH;
    
    buffer = (float *)malloc(size * sizeof(float));
    
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset_id = H5Dopen(file_id, dataset_name, H5P_DEFAULT);
    status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
    
    int idx = 0;
    for(k = 0; k < FC1_NBOUTPUT; k++) {
        for(l = 0; l < POOL2_NBOUTPUT; l++) {
            for(i = 0; i < POOL2_HEIGHT; i++) {
                for(j = 0; j < POOL2_WIDTH; j++) {
                    weights[k][l][i][j] = FLOAT_TO_FIXED(buffer[idx++]);
                }
            }
        }
    }
    
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    free(buffer);
}

/**
  * @brief  Lecture des biais FC1 avec conversion fixed-point
  */
void ReadFc1Bias_fixed(char *filename, char *dataset_name, fixed_t bias[FC1_NBOUTPUT])
{
    hid_t file_id, dataset_id;
    herr_t status;
    float *buffer;
    int i;
    
    buffer = (float *)malloc(FC1_NBOUTPUT * sizeof(float));
    
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset_id = H5Dopen(file_id, dataset_name, H5P_DEFAULT);
    status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
    
    for(i = 0; i < FC1_NBOUTPUT; i++) {
        bias[i] = FLOAT_TO_FIXED(buffer[i]);
    }
    
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    free(buffer);
}

/**
  * @brief  Lecture des poids FC2 avec conversion fixed-point
  */
void ReadFc2Weights_fixed(char *filename, char *dataset_name,
    fixed_t weights[FC2_NBOUTPUT][FC1_NBOUTPUT])
{
    hid_t file_id, dataset_id;
    herr_t status;
    float *buffer;
    int i, j;
    int size = FC2_NBOUTPUT * FC1_NBOUTPUT;
    
    buffer = (float *)malloc(size * sizeof(float));
    
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset_id = H5Dopen(file_id, dataset_name, H5P_DEFAULT);
    status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
    
    int idx = 0;
    for(i = 0; i < FC2_NBOUTPUT; i++) {
        for(j = 0; j < FC1_NBOUTPUT; j++) {
            weights[i][j] = FLOAT_TO_FIXED(buffer[idx++]);
        }
    }
    
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    free(buffer);
}

/**
  * @brief  Lecture des biais FC2 avec conversion fixed-point
  */
void ReadFc2Bias_fixed(char *filename, char *dataset_name, fixed_t bias[FC2_NBOUTPUT])
{
    hid_t file_id, dataset_id;
    herr_t status;
    float buffer[FC2_NBOUTPUT];
    int i;
    
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset_id = H5Dopen(file_id, dataset_name, H5P_DEFAULT);
    status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
    
    for(i = 0; i < FC2_NBOUTPUT; i++) {
        bias[i] = FLOAT_TO_FIXED(buffer[i]);
    }
    
    H5Dclose(dataset_id);
    H5Fclose(file_id);
}

/**
  ******************************************************************************
  * @brief   Programme principal - Inférence LeNet en fixed-point sur MNIST
  ******************************************************************************
  */
int main(void) {
    short x, y, z, k, m;
    char *hdf5_filename = "lenet_weights.weights.h5";
    
    // Noms des datasets HDF5
    char *conv1_weights = "layers/conv2d/vars/0";
    char *conv1_bias = "layers/conv2d/vars/1";
    char *conv2_weights = "layers/conv2d_1/vars/0";
    char *conv2_bias = "layers/conv2d_1/vars/1";
    char *fc1_weights = "layers/dense/vars/0";
    char *fc1_bias = "layers/dense/vars/1";
    char *fc2_weights = "layers/dense_1/vars/0";
    char *fc2_bias = "layers/dense_1/vars/1";
    
    char *test_labels_filename = "mnist/t10k-labels-idx1-ubyte";
    FILE *label_file;
    int ret;
    unsigned char label, number;
    unsigned int error;
    unsigned char labels_legend[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    char img_filename[120];
    char img_count[10];
    fixed_t max;
    struct timeval start, end;
    double tdiff;
    
    printf("\e[1;1H\e[2J");
    printf("\n=================================================\n");
    printf("LeNet CNN - FIXED-POINT VERSION (Q4.12 format)\n");
    printf("=================================================\n");
    printf("Format: %d bits entiers, %d bits fractionnaires\n", 
           16 - FIXED_POINT_FRACTIONAL_BITS, FIXED_POINT_FRACTIONAL_BITS);
    printf("Type: int16_t, Plage: [-8.0, +7.999]\n");
    printf("Précision: ~%.6f\n", 1.0 / FIXED_POINT_MULTIPLIER);
    
    printf("\nChargement des poids (conversion float->fixed)...\n");
    ReadConv1Weights_fixed(hdf5_filename, conv1_weights, CONV1_KERNEL);
    printf("  Conv1 weights: OK\n");
    ReadConv1Bias_fixed(hdf5_filename, conv1_bias, CONV1_BIAS);
    printf("  Conv1 bias: OK\n");
    ReadConv2Weights_fixed(hdf5_filename, conv2_weights, CONV2_KERNEL);
    printf("  Conv2 weights: OK\n");
    ReadConv2Bias_fixed(hdf5_filename, conv2_bias, CONV2_BIAS);
    printf("  Conv2 bias: OK\n");
    ReadFc1Weights_fixed(hdf5_filename, fc1_weights, FC1_KERNEL);
    printf("  FC1 weights: OK\n");
    ReadFc1Bias_fixed(hdf5_filename, fc1_bias, FC1_BIAS);
    printf("  FC1 bias: OK\n");
    ReadFc2Weights_fixed(hdf5_filename, fc2_weights, FC2_KERNEL);
    printf("  FC2 weights: OK\n");
    ReadFc2Bias_fixed(hdf5_filename, fc2_bias, FC2_BIAS);
    printf("  FC2 bias: OK\n");
    printf("Tous les poids chargés avec succès!\n");
    
    printf("\nOuverture du fichier de labels...\n");
    label_file = fopen(test_labels_filename, "r");
    if (!label_file) {
        printf("Erreur: Impossible d'ouvrir %s\n", test_labels_filename);
        exit(1);
    }
    
    // Skip 8 header bytes
    for (k = 0; k < 8; k++)
        ret = fscanf(label_file, "%c", &label);
    
    printf("\nDémarrage du traitement...\n");
    printf("=================================================\n\n");
    
    m = 0;          // Compteur d'images
    error = 0;      // Nombre d'erreurs
    
    // BOUCLE PRINCIPALE DE TEST
    gettimeofday(&start, NULL);
    
    while (1) {
        ret = fscanf(label_file, "%c", &label);
        if (feof(label_file)) break;
        
        // Construction du nom de fichier
        strcpy(img_filename, "mnist/t10k-images-idx3-ubyte[");
        sprintf(img_count, "%d", m);
        if      (m < 10)    strcat(img_filename, "0000");
        else if (m < 100)   strcat(img_filename, "000");
        else if (m < 1000)  strcat(img_filename, "00");
        else if (m < 10000) strcat(img_filename, "0");
        strcat(img_filename, img_count);
        strcat(img_filename, "].pgm");
        
        printf("\033[%d;%dH%s\n", 7, 0, img_filename);
        
        // Lecture de l'image
        ReadPgmFile(img_filename, (unsigned char *)REF_IMG);
        
        // Normalisation avec conversion fixed-point
        NormalizeImg_fixed((unsigned char *)REF_IMG, (fixed_t *)INPUT_NORM, 
                          IMG_WIDTH, IMG_HEIGHT);
        
        // Inférence CNN en fixed-point
        lenet_cnn_fixed(INPUT_NORM,
                       CONV1_KERNEL,
                       CONV1_BIAS,
                       CONV2_KERNEL,
                       CONV2_BIAS,
                       FC1_KERNEL,
                       FC1_BIAS,
                       FC2_KERNEL,
                       FC2_BIAS,
                       FC2_OUTPUT);
        
        // Softmax
        Softmax_fixed(FC2_OUTPUT, SOFTMAX_OUTPUT);
        
        // Affichage des probabilités
        printf("\nSortie Softmax (fixed-point):\n");
        max = 0;
        number = 0;
        for (k = 0; k < FC2_NBOUTPUT; k++) {
            float prob = FIXED_TO_FLOAT(SOFTMAX_OUTPUT[k]) * 100.0f;
            printf("%.2f%% ", prob);
            if (SOFTMAX_OUTPUT[k] > max) {
                max = SOFTMAX_OUTPUT[k];
                number = k;
            }
        }
        
        // Résultat
        printf("\n\nPrédiction: %d \t Réel: %d ", labels_legend[number], label);
        if (labels_legend[number] != label) {
            printf("[ERREUR]");
            error++;
        } else {
            printf("[✓ OK]");
        }
        printf("\n");
        
        m++;
        
        // Affichage de progression tous les 100 images
        if (m % 100 == 0) {
            float current_accuracy = (1.0f - ((float)error / m)) * 100.0f;
            printf("\n--- Progression: %d images, Précision actuelle: %.2f%% ---\n\n", 
                   m, current_accuracy);
        }
        
        // Arrêt optionnel après N images (pour debug)
        // if (m >= 100) break;
    }
    
    gettimeofday(&end, NULL);
    
    // RÉSULTATS FINAUX
    printf("\n\n");
    printf("=================================================\n");
    printf("         RÉSULTATS FINAUX - FIXED-POINT         \n");
    printf("=================================================\n");
    
    tdiff = (double)(end.tv_sec - start.tv_sec) + 
            (double)(end.tv_usec - start.tv_usec) / 1000000.0;
    
    printf("\nStatistiques:\n");
    printf("  Images traitées:     %d\n", m);
    printf("  Prédictions justes:  %d\n", m - error);
    printf("  Erreurs:             %d\n", error);
    printf("\nPerformance:\n");
    printf("  Taux de réussite:    %.2f%%\n", (1.0f - ((float)error / m)) * 100.0f);
    printf("  Temps total:         %.3f s\n", tdiff);
    printf("  Temps moyen/image:   %.3f ms\n", (tdiff / m) * 1000.0);
    printf("  Throughput:          %.1f images/s\n", m / tdiff);
    printf("\nFormat Fixed-Point:\n");
    printf("  Type:                Q%d.%d (int16_t)\n", 
           16 - FIXED_POINT_FRACTIONAL_BITS, FIXED_POINT_FRACTIONAL_BITS);
    printf("  Taille mémoire:      50%% de la version float\n");
    
    printf("\n=================================================\n\n");
    
    fclose(label_file);
    
    return 0;
}
