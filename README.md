
# Pediatric brain tumor classification using deep learning on MR-images with age fusion

This repository contains code to support the study of deep learning-based classification of pediatric brain tumors using MRI data from the CBTN dataset.

[Preprint](https://doi.org/10.1101/2024.09.05.24313109) | [Cite](#reference)


**Abstract**

**Purpose**:To implement and evaluate deep learning-based methods for the classification of pediatric brain tumors in MR data.

**Materials and methods**: A subset of the “Children’s Brain Tumor Network” dataset was retrospectively used (n=178 subjects, female=72, male=102, NA=4, age-range [0.01, 36.49] years) with tumor types being low-grade astrocytoma (n=84), ependymoma (n=32), and medulloblastoma (n=62). T1w post-contrast (n=94 subjects), T2w (n=160 subjects), and ADC (n=66 subjects) MR sequences were used separately. Two deep-learning models were trained on transversal slices showing tumor. Joint fusion was implemented to combine image and age data, and two pre-training paradigms were utilized. Model explainability was investigated using gradient-weighted class activation mapping (Grad-CAM), and the learned feature space was visualized using principal component analysis (PCA).

**Results**: The highest tumor-type classification performance was achieved when using a vision transformer model pre-trained on ImageNet and fine-tuned on ADC images with age fusion (MCC: 0.77 ± 0.14 Accuracy: 0.87 ± 0.08), followed by models trained on T2w (MCC: 0.58 ± 0.11, Accuracy: 0.73 ± 0.08) and T1w post-contrast (MCC: 0.41 ± 0.11, Accuracy: 0.62 ± 0.08) data. Age fusion marginally improved the model’s performance. Both model architectures performed similarly across the experiments, with no differences between the pre-training strategies. Grad-CAMs showed that the models’ attention focused on the brain region. PCA of the feature space showed greater separation of the tumor-type clusters when using contrastive pre-training.

**Conclusion**: Classification of pediatric brain tumors on MR-images could be accomplished using deep learning, with the top-performing model being trained on ADC data, which is used by radiologists for the clinical classification of these tumors.
Key pointsThe vision transformer model pre-trained on ImageNet and fine-tuned on ADC data with age fusion achieved the highest performance, which was significantly better than models trained on T2w (second-best) and T1w-Gd data.Fusion of age information with the image data marginally improved classification, and model architecture (ResNet50 -vs -ViT) and pre-training strategies (supervised -vs -self-supervised) did not show to significantly impact models’ performance.Model explainability, by means of class activation mapping and principal component analysis of the learned feature space, show that the models use the tumor region information for classification and that the tumor type clusters are better separated when using age information.
Summary Deep learning-based classification of pediatric brain tumors can be achieved using single-sequence pre-operative MR data, showing the potential of automated decision support tools that can aid radiologists in the primary diagnosis of these tumors.

**Key highlights:**

- **The vision transformer model** pre-trained on ImageNet and fine-tuned on ADC data with age fusion achieved the highest performance, which was significantly better than models trained on T2w (second-best) and T1w-Gd data.
- **Fusion of age information** with the image data marginally improved classification, and model architecture (ResNet50 -vs -ViT) and pre-training strategies (supervised -vs -self-supervised) did not show to significant impact on models’ performance.
- **Model explainability**, by means of class activation mapping and principal component analysis of the learned feature space, show that the models use the tumor region information for classification and that the tumor type clusters are better separated when using age information.

## Table of Contents
- [Setup](#Setup)
- [Datasets](#datasets)
- [Code Structure](#code-structure)
- [Usage](#usage)
- [Reference](#reference)
- [License](#license)
---
## Setup
TODO. 

## Dataset
The dataset used for this project was obtained from [CBTN](https://cbtn.org/). 

## Code structure
Model pretraining, fine-tuning and evaluation are run using .py scripts using hydra configuration files. The configuration files for contrastive pretraining (``SimCLR_config.yaml``), model classification training (``config.yaml``) and evaluation (``evaluation_config.yaml``) can be found in the conf folder. 
Additionally, configuration files specifying the dataset settings are available in the config/dataset folder. See the different configuration files for all the available tunable settings for model pretraining and classification training.

## Usage
- **Model pretraining**: model pretraining is run though the ``run_SimCLR_preptraining.py`` script which is configured with the ``SimCLR_config.yaml`` configuration file. After setting the appropriate paths in the configuration file, run the code below for model pretraining:
```bash 
python3 run_SimCLR_pretraining.py 
```
- **Classifier training**: training of the classification model (from scratch of fine-tuned) can be run using the ``run_classification_training.py`` script. This is configured using the (``config.yaml``). After setting the appropriate paths and parameters, run the code below for training the classifier:
```bash 
python3 run_classification_training.py
```
- **Model evaluation**: model evaluation can be obtained using the ``run_model_evaluate.py`` script, which is configured using (``evaluation_config.yaml``). This script generates a tabular .csv file which can be aggregated across several classification model configurations using the ``aggregate_evaluation_csv_files.py`` script. 
A summary of the test evaluation can be printed on display using the ``print_aggregated_evaluation_summary.py`` and plots of the different metrics can be obtained using ``plot_aggregated_evaluation_summary.py``.
- **Grad-CAMs**: model explainability maps through Grad-CAM can be obtained using the ``apply_gradCam.py`` script. See the comments in the script on how to set the different model and dataset paths.

## Reference
If you use this work, please cite:

```bibtex
@misc{tampu_pediatric_2024,
	title = {Pediatric brain tumor classification using deep learning on {MR}-images with age fusion},
	url = {https://www.medrxiv.org/content/10.1101/2024.09.05.24313109v1},
	doi = {10.1101/2024.09.05.24313109},
	publisher = {medRxiv},
	author = {Tampu, Iulian Emil and Bianchessi, Tamara and Blystad, Ida and Lundberg, Peter and Nyman, Per and Eklund, Anders and Haj-Hosseini, Neda},
	month = sep,
	year = {2024},
}
```

## License
This work is licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/).
