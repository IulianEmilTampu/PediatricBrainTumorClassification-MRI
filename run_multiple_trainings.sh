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
IMG_DATASET_FOLDER=/flush/iulta54/Research/Data/CBTN/EXTRACTED_SLICES_TFR
DATASET_TYPE="CBTN"
NBR_CLASSES=3
GPU_NBR=$gpu
NBR_FOLDS=1
LEARNING_RATE=0.0001
BATCH_SIZE=32
MAX_EPOCHS=75
USE_PRETRAINED_MODEL=False
LOSS="CCE"
RANDOM_SEED_NUMBER=29122009
MR_MODALITIES="T2"
DEBUG_DATASET_FRACTION=1
TFR_DATA=True
MODEL_NAME="ClassificationModel"

# ##############
echo "##################################### RUNNING TRAININGS WITHOUT AGE AND WITHOUT GRADCAM"
# ##############
USE_AGE=False
USE_GRADCAM=False
MODEL_TYPE="SDM4"
OPTIMIZER="ADAM"
# RUN HYPER PARAMETER SHEARCH
for LEARNING_RATE in 0.0001 
do
   for LOSS in "CCE" "MCC_and_CCE_Loss" 
   do
      for OPTIMIZER in 'ADAM'
      do
         python3 run_model_training_TF.py --WORKING_FOLDER $WORKING_FOLDER --IMG_DATASET_FOLDER $IMG_DATASET_FOLDER --DATASET_TYPE $DATASET_TYPE --NBR_CLASSES $NBR_CLASSES --GPU_NBR $GPU_NBR --NBR_FOLDS $NBR_FOLDS --LEARNING_RATE $LEARNING_RATE --BATCH_SIZE $BATCH_SIZE --MAX_EPOCHS $MAX_EPOCHS --USE_AGE $USE_AGE --USE_GRADCAM $USE_GRADCAM --LOSS $LOSS --RANDOM_SEED_NUMBER $RANDOM_SEED_NUMBER --MR_MODALITIES $MR_MODALITIES --DEBUG_DATASET_FRACTION $DEBUG_DATASET_FRACTION --TFR_DATA $TFR_DATA --MODEL_TYPE $MODEL_TYPE --MODEL_NAME $MODEL_NAME --OPTIMIZER $OPTIMIZER
      done
   done
done

USE_AGE=True
USE_GRADCAM=False
MODEL_TYPE="SDM4"
OPTIMIZER="ADAM"
# RUN HYPER PARAMETER SHEARCH
for LEARNING_RATE in 0.0001 
do
   for LOSS in "CCE" "MCC_and_CCE_Loss" 
   do
      for OPTIMIZER in 'ADAM'
      do
         python3 run_model_training_TF.py --WORKING_FOLDER $WORKING_FOLDER --IMG_DATASET_FOLDER $IMG_DATASET_FOLDER --DATASET_TYPE $DATASET_TYPE --NBR_CLASSES $NBR_CLASSES --GPU_NBR $GPU_NBR --NBR_FOLDS $NBR_FOLDS --LEARNING_RATE $LEARNING_RATE --BATCH_SIZE $BATCH_SIZE --MAX_EPOCHS $MAX_EPOCHS --USE_AGE $USE_AGE --USE_GRADCAM $USE_GRADCAM --LOSS $LOSS --RANDOM_SEED_NUMBER $RANDOM_SEED_NUMBER --MR_MODALITIES $MR_MODALITIES --DEBUG_DATASET_FRACTION $DEBUG_DATASET_FRACTION --TFR_DATA $TFR_DATA --MODEL_TYPE $MODEL_TYPE --MODEL_NAME $MODEL_NAME --OPTIMIZER $OPTIMIZER
      done
   done
done

USE_AGE=False
USE_GRADCAM=True
MODEL_TYPE="SDM4"
OPTIMIZER="ADAM"
# RUN HYPER PARAMETER SHEARCH
for LEARNING_RATE in 0.0001 
do
   for LOSS in "CCE" "MCC_and_CCE_Loss" 
   do
      for OPTIMIZER in 'ADAM'
      do
         python3 run_model_training_TF.py --WORKING_FOLDER $WORKING_FOLDER --IMG_DATASET_FOLDER $IMG_DATASET_FOLDER --DATASET_TYPE $DATASET_TYPE --NBR_CLASSES $NBR_CLASSES --GPU_NBR $GPU_NBR --NBR_FOLDS $NBR_FOLDS --LEARNING_RATE $LEARNING_RATE --BATCH_SIZE $BATCH_SIZE --MAX_EPOCHS $MAX_EPOCHS --USE_AGE $USE_AGE --USE_GRADCAM $USE_GRADCAM --LOSS $LOSS --RANDOM_SEED_NUMBER $RANDOM_SEED_NUMBER --MR_MODALITIES $MR_MODALITIES --DEBUG_DATASET_FRACTION $DEBUG_DATASET_FRACTION --TFR_DATA $TFR_DATA --MODEL_TYPE $MODEL_TYPE --MODEL_NAME $MODEL_NAME --OPTIMIZER $OPTIMIZER
      done
   done
done

USE_AGE=True
USE_GRADCAM=True
MODEL_TYPE="SDM4"
OPTIMIZER="ADAM"
# RUN HYPER PARAMETER SHEARCH
for LEARNING_RATE in 0.0001 
do
   for LOSS in "CCE" "MCC_and_CCE_Loss" 
   do
      for OPTIMIZER in 'ADAM'
      do
         python3 run_model_training_TF.py --WORKING_FOLDER $WORKING_FOLDER --IMG_DATASET_FOLDER $IMG_DATASET_FOLDER --DATASET_TYPE $DATASET_TYPE --NBR_CLASSES $NBR_CLASSES --GPU_NBR $GPU_NBR --NBR_FOLDS $NBR_FOLDS --LEARNING_RATE $LEARNING_RATE --BATCH_SIZE $BATCH_SIZE --MAX_EPOCHS $MAX_EPOCHS --USE_AGE $USE_AGE --USE_GRADCAM $USE_GRADCAM --LOSS $LOSS --RANDOM_SEED_NUMBER $RANDOM_SEED_NUMBER --MR_MODALITIES $MR_MODALITIES --DEBUG_DATASET_FRACTION $DEBUG_DATASET_FRACTION --TFR_DATA $TFR_DATA --MODEL_TYPE $MODEL_TYPE --MODEL_NAME $MODEL_NAME --OPTIMIZER $OPTIMIZER
      done
   done
done

# # ##############
# # ##############
# USE_AGE=False
# USE_GRADCAM=False
# MODEL_TYPE="ViT"
# OPTIMIZER="ADAM"

# python3 run_model_training_TF.py --WORKING_FOLDER $WORKING_FOLDER --IMG_DATASET_FOLDER $IMG_DATASET_FOLDER --DATASET_TYPE $DATASET_TYPE --NBR_CLASSES $NBR_CLASSES --GPU_NBR $GPU_NBR --NBR_FOLDS $NBR_FOLDS --LEARNING_RATE $LEARNING_RATE --BATCH_SIZE $BATCH_SIZE --MAX_EPOCHS $MAX_EPOCHS --USE_AGE $USE_AGE --USE_GRADCAM $USE_GRADCAM --LOSS $LOSS --RANDOM_SEED_NUMBER $RANDOM_SEED_NUMBER --MR_MODALITIES $MR_MODALITIES --DEBUG_DATASET_FRACTION $DEBUG_DATASET_FRACTION --TFR_DATA $TFR_DATA --MODEL_TYPE $MODEL_TYPE --MODEL_NAME $MODEL_NAME --OPTIMIZER $OPTIMIZER

# # ##############
# # ##############
# USE_AGE=False
# USE_GRADCAM=False
# MODEL_TYPE="ResNet9"
# OPTIMIZER="ADAM"

# python3 run_model_training_TF.py --WORKING_FOLDER $WORKING_FOLDER --IMG_DATASET_FOLDER $IMG_DATASET_FOLDER --DATASET_TYPE $DATASET_TYPE --NBR_CLASSES $NBR_CLASSES --GPU_NBR $GPU_NBR --NBR_FOLDS $NBR_FOLDS --LEARNING_RATE $LEARNING_RATE --BATCH_SIZE $BATCH_SIZE --MAX_EPOCHS $MAX_EPOCHS --USE_AGE $USE_AGE --USE_GRADCAM $USE_GRADCAM --LOSS $LOSS --RANDOM_SEED_NUMBER $RANDOM_SEED_NUMBER --MR_MODALITIES $MR_MODALITIES --DEBUG_DATASET_FRACTION $DEBUG_DATASET_FRACTION --TFR_DATA $TFR_DATA --MODEL_TYPE $MODEL_TYPE --MODEL_NAME $MODEL_NAME --OPTIMIZER $OPTIMIZER

# # ##############
# echo "##################################### RUNNING TRAININGS WITH AGE AND WITHOUT GRADCAM"
# # ##############
# USE_AGE=True
# USE_GRADCAM=False
# MODEL_TYPE="SDM4"
# OPTIMIZER="ADAM"

# python3 run_model_training_TF.py --WORKING_FOLDER $WORKING_FOLDER --IMG_DATASET_FOLDER $IMG_DATASET_FOLDER --DATASET_TYPE $DATASET_TYPE --NBR_CLASSES $NBR_CLASSES --GPU_NBR $GPU_NBR --NBR_FOLDS $NBR_FOLDS --LEARNING_RATE $LEARNING_RATE --BATCH_SIZE $BATCH_SIZE --MAX_EPOCHS $MAX_EPOCHS --USE_AGE $USE_AGE --USE_GRADCAM $USE_GRADCAM --LOSS $LOSS --RANDOM_SEED_NUMBER $RANDOM_SEED_NUMBER --MR_MODALITIES $MR_MODALITIES --DEBUG_DATASET_FRACTION $DEBUG_DATASET_FRACTION --TFR_DATA $TFR_DATA --MODEL_TYPE $MODEL_TYPE --MODEL_NAME $MODEL_NAME --OPTIMIZER $OPTIMIZER

# # # ##############
# # # ##############
# # USE_AGE=True
# # USE_GRADCAM=False
# # MODEL_TYPE="ViT"
# # OPTIMIZER="ADAM"

# # python3 run_model_training_TF.py --WORKING_FOLDER $WORKING_FOLDER --IMG_DATASET_FOLDER $IMG_DATASET_FOLDER --DATASET_TYPE $DATASET_TYPE --NBR_CLASSES $NBR_CLASSES --GPU_NBR $GPU_NBR --NBR_FOLDS $NBR_FOLDS --LEARNING_RATE $LEARNING_RATE --BATCH_SIZE $BATCH_SIZE --MAX_EPOCHS $MAX_EPOCHS --USE_AGE $USE_AGE --USE_GRADCAM $USE_GRADCAM --LOSS $LOSS --RANDOM_SEED_NUMBER $RANDOM_SEED_NUMBER --MR_MODALITIES $MR_MODALITIES --DEBUG_DATASET_FRACTION $DEBUG_DATASET_FRACTION --TFR_DATA $TFR_DATA --MODEL_TYPE $MODEL_TYPE --MODEL_NAME $MODEL_NAME --OPTIMIZER $OPTIMIZER

# # # ##############
# # # ##############
# # USE_AGE=True
# # USE_GRADCAM=False
# # MODEL_TYPE="ResNet9"
# # OPTIMIZER="ADAM"

# # python3 run_model_training_TF.py --WORKING_FOLDER $WORKING_FOLDER --IMG_DATASET_FOLDER $IMG_DATASET_FOLDER --DATASET_TYPE $DATASET_TYPE --NBR_CLASSES $NBR_CLASSES --GPU_NBR $GPU_NBR --NBR_FOLDS $NBR_FOLDS --LEARNING_RATE $LEARNING_RATE --BATCH_SIZE $BATCH_SIZE --MAX_EPOCHS $MAX_EPOCHS --USE_AGE $USE_AGE --USE_GRADCAM $USE_GRADCAM --LOSS $LOSS --RANDOM_SEED_NUMBER $RANDOM_SEED_NUMBER --MR_MODALITIES $MR_MODALITIES --DEBUG_DATASET_FRACTION $DEBUG_DATASET_FRACTION --TFR_DATA $TFR_DATA --MODEL_TYPE $MODEL_TYPE --MODEL_NAME $MODEL_NAME --OPTIMIZER $OPTIMIZER

# # ##############
# echo "##################################### RUNNING TRAININGS WITHOUT AGE AND WITH GRADCAM"
# # ##############
# USE_AGE=False
# USE_GRADCAM=True
# MODEL_TYPE="SDM4"
# OPTIMIZER="ADAM"

# python3 run_model_training_TF.py --WORKING_FOLDER $WORKING_FOLDER --IMG_DATASET_FOLDER $IMG_DATASET_FOLDER --DATASET_TYPE $DATASET_TYPE --NBR_CLASSES $NBR_CLASSES --GPU_NBR $GPU_NBR --NBR_FOLDS $NBR_FOLDS --LEARNING_RATE $LEARNING_RATE --BATCH_SIZE $BATCH_SIZE --MAX_EPOCHS $MAX_EPOCHS --USE_AGE $USE_AGE --USE_GRADCAM $USE_GRADCAM --LOSS $LOSS --RANDOM_SEED_NUMBER $RANDOM_SEED_NUMBER --MR_MODALITIES $MR_MODALITIES --DEBUG_DATASET_FRACTION $DEBUG_DATASET_FRACTION --TFR_DATA $TFR_DATA --MODEL_TYPE $MODEL_TYPE --MODEL_NAME $MODEL_NAME --OPTIMIZER $OPTIMIZER

# # # ##############
# # # ##############
# # USE_AGE=False
# # USE_GRADCAM=True
# # MODEL_TYPE="ViT"
# # OPTIMIZER="ADAM"

# # python3 run_model_training_TF.py --WORKING_FOLDER $WORKING_FOLDER --IMG_DATASET_FOLDER $IMG_DATASET_FOLDER --DATASET_TYPE $DATASET_TYPE --NBR_CLASSES $NBR_CLASSES --GPU_NBR $GPU_NBR --NBR_FOLDS $NBR_FOLDS --LEARNING_RATE $LEARNING_RATE --BATCH_SIZE $BATCH_SIZE --MAX_EPOCHS $MAX_EPOCHS --USE_AGE $USE_AGE --USE_GRADCAM $USE_GRADCAM --LOSS $LOSS --RANDOM_SEED_NUMBER $RANDOM_SEED_NUMBER --MR_MODALITIES $MR_MODALITIES --DEBUG_DATASET_FRACTION $DEBUG_DATASET_FRACTION --TFR_DATA $TFR_DATA --MODEL_TYPE $MODEL_TYPE --MODEL_NAME $MODEL_NAME --OPTIMIZER $OPTIMIZER

# # # ##############
# # # ##############
# # USE_AGE=False
# # USE_GRADCAM=True
# # MODEL_TYPE="ResNet9"
# # OPTIMIZER="ADAM"

# # python3 run_model_training_TF.py --WORKING_FOLDER $WORKING_FOLDER --IMG_DATASET_FOLDER $IMG_DATASET_FOLDER --DATASET_TYPE $DATASET_TYPE --NBR_CLASSES $NBR_CLASSES --GPU_NBR $GPU_NBR --NBR_FOLDS $NBR_FOLDS --LEARNING_RATE $LEARNING_RATE --BATCH_SIZE $BATCH_SIZE --MAX_EPOCHS $MAX_EPOCHS --USE_AGE $USE_AGE --USE_GRADCAM $USE_GRADCAM --LOSS $LOSS --RANDOM_SEED_NUMBER $RANDOM_SEED_NUMBER --MR_MODALITIES $MR_MODALITIES --DEBUG_DATASET_FRACTION $DEBUG_DATASET_FRACTION --TFR_DATA $TFR_DATA --MODEL_TYPE $MODEL_TYPE --MODEL_NAME $MODEL_NAME --OPTIMIZER $OPTIMIZER


# # ##############
# echo "##################################### RUNNING TRAININGS WITH AGE AND WITH GRADCAM"
# # ##############
# USE_AGE=True
# USE_GRADCAM=True
# MODEL_TYPE="SDM4"
# OPTIMIZER="ADAM"

# python3 run_model_training_TF.py --WORKING_FOLDER $WORKING_FOLDER --IMG_DATASET_FOLDER $IMG_DATASET_FOLDER --DATASET_TYPE $DATASET_TYPE --NBR_CLASSES $NBR_CLASSES --GPU_NBR $GPU_NBR --NBR_FOLDS $NBR_FOLDS --LEARNING_RATE $LEARNING_RATE --BATCH_SIZE $BATCH_SIZE --MAX_EPOCHS $MAX_EPOCHS --USE_AGE $USE_AGE --USE_GRADCAM $USE_GRADCAM --LOSS $LOSS --RANDOM_SEED_NUMBER $RANDOM_SEED_NUMBER --MR_MODALITIES $MR_MODALITIES --DEBUG_DATASET_FRACTION $DEBUG_DATASET_FRACTION --TFR_DATA $TFR_DATA --MODEL_TYPE $MODEL_TYPE --MODEL_NAME $MODEL_NAME --OPTIMIZER $OPTIMIZER

# # # ##############
# # # ##############
# # USE_AGE=True
# # USE_GRADCAM=True
# # MODEL_TYPE="ViT"
# # OPTIMIZER="ADAM"

# # python3 run_model_training_TF.py --WORKING_FOLDER $WORKING_FOLDER --IMG_DATASET_FOLDER $IMG_DATASET_FOLDER --DATASET_TYPE $DATASET_TYPE --NBR_CLASSES $NBR_CLASSES --GPU_NBR $GPU_NBR --NBR_FOLDS $NBR_FOLDS --LEARNING_RATE $LEARNING_RATE --BATCH_SIZE $BATCH_SIZE --MAX_EPOCHS $MAX_EPOCHS --USE_AGE $USE_AGE --USE_GRADCAM $USE_GRADCAM --LOSS $LOSS --RANDOM_SEED_NUMBER $RANDOM_SEED_NUMBER --MR_MODALITIES $MR_MODALITIES --DEBUG_DATASET_FRACTION $DEBUG_DATASET_FRACTION --TFR_DATA $TFR_DATA --MODEL_TYPE $MODEL_TYPE --MODEL_NAME $MODEL_NAME --OPTIMIZER $OPTIMIZER

# # # ##############
# # # ##############
# # USE_AGE=True
# # USE_GRADCAM=True
# # MODEL_TYPE="ResNet9"
# # OPTIMIZER="ADAM"

# # python3 run_model_training_TF.py --WORKING_FOLDER $WORKING_FOLDER --IMG_DATASET_FOLDER $IMG_DATASET_FOLDER --DATASET_TYPE $DATASET_TYPE --NBR_CLASSES $NBR_CLASSES --GPU_NBR $GPU_NBR --NBR_FOLDS $NBR_FOLDS --LEARNING_RATE $LEARNING_RATE --BATCH_SIZE $BATCH_SIZE --MAX_EPOCHS $MAX_EPOCHS --USE_AGE $USE_AGE --USE_GRADCAM $USE_GRADCAM --LOSS $LOSS --RANDOM_SEED_NUMBER $RANDOM_SEED_NUMBER --MR_MODALITIES $MR_MODALITIES --DEBUG_DATASET_FRACTION $DEBUG_DATASET_FRACTION --TFR_DATA $TFR_DATA --MODEL_TYPE $MODEL_TYPE --MODEL_NAME $MODEL_NAME --OPTIMIZER $OPTIMIZER






