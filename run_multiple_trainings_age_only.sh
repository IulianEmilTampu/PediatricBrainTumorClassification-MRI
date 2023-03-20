#!/bin/bash

Help()
{
   # Display Help
   echo "Bash script that runs the training routine in the context of the MICCAI 2023 project"
   echo
   echo "Syntax: run_training [w|d|g]"
   echo "required inputs:"
   echo "w     Working folder (where the scripts are)"
   echo "g     GPU number on which to run testing"
   echo
}

while getopts w:hm:g:d: option; do
case "${option}" in
   h) # display Help
       Help
       exit;;
   w) working_folder=${OPTARG};;
   d) dataset_folder=${OPTARG};;
   g) gpu=${OPTARG};;

   \?) # incorrect option
         echo "Error: Invalid input"
         exit 1
esac
done

# make sure to have the right conda environment open when running the script
# activate conda environment
eval "$(conda shell.bash hook)"
conda activate MICCAI2023_TF

# work on GPU 0
export CUDA_VISIBLE_DEVICES=$gpu

# go to the working folder
cd $working_folder

# #################
# DEFINE PARAMETERS
# #################
WORKING_FOLDER=$working_folder
IMG_DATASET_FOLDER=/flush/iulta54/Research/Data/CBTN/EXTRACTED_SLICES_TFR_MERGED_FROM_TB_20230320
DATASET_TYPE="CBTN"
NBR_CLASSES=3
GPU_NBR=$gpu
NBR_FOLDS=5
DATA_NORMALIZATION=True

OPTIMIZER="ADAM"
LEARNING_RATE=0.01
LOSS="CCE"

BATCH_SIZE=8
MAX_EPOCHS=50

DEBUG_DATASET_FRACTION=1

# # ##############
# echo "######  RUNNING TRAINING AGE_TO_CLASSES  ######"
# # ##############

# MODEL_NAME="Age_only_classification"
# MODEL_VERSION=age_to_classes


# for RANDOM_SEED_NUMBER in 1111 1112
# do
#    python3 run_model_training_age_only.py --WORKING_FOLDER $WORKING_FOLDER --IMG_DATASET_FOLDER $IMG_DATASET_FOLDER --DATASET_TYPE $DATASET_TYPE --NBR_CLASSES $NBR_CLASSES --GPU_NBR $GPU_NBR --NBR_FOLDS $NBR_FOLDS --LEARNING_RATE $LEARNING_RATE --BATCH_SIZE $BATCH_SIZE --MAX_EPOCHS $MAX_EPOCHS --LOSS $LOSS --RANDOM_SEED_NUMBER $RANDOM_SEED_NUMBER --DEBUG_DATASET_FRACTION $DEBUG_DATASET_FRACTION --MODEL_VERSION $MODEL_VERSION --MODEL_NAME $MODEL_NAME --OPTIMIZER $OPTIMIZER --DATA_NORMALIZATION $DATA_NORMALIZATION
# done

# ##############
echo "######  RUNNING TRAINING SIMPLE_AGE_ENCODER  ######"
# ##############

MODEL_NAME="Age_only_classification"
MODEL_VERSION=simple_age_encoder

for RANDOM_SEED_NUMBER in 1111 1112
do
   python3 run_model_training_age_only.py --WORKING_FOLDER $WORKING_FOLDER --IMG_DATASET_FOLDER $IMG_DATASET_FOLDER --DATASET_TYPE $DATASET_TYPE --NBR_CLASSES $NBR_CLASSES --GPU_NBR $GPU_NBR --NBR_FOLDS $NBR_FOLDS --LEARNING_RATE $LEARNING_RATE --BATCH_SIZE $BATCH_SIZE --MAX_EPOCHS $MAX_EPOCHS --LOSS $LOSS --RANDOM_SEED_NUMBER $RANDOM_SEED_NUMBER --DEBUG_DATASET_FRACTION $DEBUG_DATASET_FRACTION --MODEL_VERSION $MODEL_VERSION --MODEL_NAME $MODEL_NAME --OPTIMIZER $OPTIMIZER --DATA_NORMALIZATION $DATA_NORMALIZATION
done

# ##############
echo "######  RUNNING TRAINING LARGE_AGE_ENCODER  ######"
# ##############

MODEL_NAME="Age_only_classification"
MODEL_VERSION=large_age_encoder

for RANDOM_SEED_NUMBER in 1111 1112
do
   python3 run_model_training_age_only.py --WORKING_FOLDER $WORKING_FOLDER --IMG_DATASET_FOLDER $IMG_DATASET_FOLDER --DATASET_TYPE $DATASET_TYPE --NBR_CLASSES $NBR_CLASSES --GPU_NBR $GPU_NBR --NBR_FOLDS $NBR_FOLDS --LEARNING_RATE $LEARNING_RATE --BATCH_SIZE $BATCH_SIZE --MAX_EPOCHS $MAX_EPOCHS --LOSS $LOSS --RANDOM_SEED_NUMBER $RANDOM_SEED_NUMBER --DEBUG_DATASET_FRACTION $DEBUG_DATASET_FRACTION --MODEL_VERSION $MODEL_VERSION --MODEL_NAME $MODEL_NAME --OPTIMIZER $OPTIMIZER --DATA_NORMALIZATION $DATA_NORMALIZATION
done