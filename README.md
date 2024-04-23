# Interpretable-through-prototypes deepfake detection for diffusion models

*Paper*: Aghasanli, Agil, Dmitry Kangin, and Plamen Angelov. "Interpretable-through-prototypes deepfake detection for diffusion models." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.
https://openaccess.thecvf.com/content/ICCV2023W/DFAD/papers/Aghasanli_Interpretable-Through-Prototypes_Deepfake_Detection_for_Diffusion_Models_ICCVW_2023_paper.pdf

*Dataset*: https://www.research.lancs.ac.uk/portal/en/datasets/interpretablethroughprototypes-deepfake-detection-for-diffusion-models(591db8a2-5e54-424c-b22b-7a4ccc8eea4c).html

## Getting Started

### Prerequisites
Ensure you have downloaded the project datasets before proceeding with the execution of any script.

## Feature Extraction

### Using Pretrained Models
1. Open the `Feature_extraction_pretrained.py` file.
2. Set the `data_dir` variable in line 72 to the appropriate path on your local machine where the datasets are stored.
3. Run the script using the command:
```
python Feature_extraction_pretrained.py
```
This script will generate four CSV files containing the train/test features and labels.

### Using Finetuned Weights
1. Open the `Feature_extraction_finetuned.py` file.
2. Adjust the `data_dir` variable in line 66 to match the path on your local machine.
3. In line 25, initialize the ViT model using the appropriate .bin and .json files generated from the finetuning process (refer to the Finetuning section below).
4. Execute the script with:
```
python Feature_extraction_finetuned.py
```
Similar to the pretrained model, this will generate four distinct CSV files for train/test features and labels.

## Classifier Testing

### xDNN Classifier
After obtaining the necessary CSV files:
1. After generating the CSV files, update lines 13-16 in `xDNN_run.py` to import the correct CSV files.
2. Run the `xDNN_run.py` script to test the xDNN classifier:
```
python xDNN_run.py
```

This script also generates a data file containing prototypes for later use (e.g., explainability).

### Other Classifiers
To see the results using SVM, KNN, and Naive Bayes:
1. Update lines 15-18 in `test_classifiers.py` to be compatible with the names of the generated CSV files.
2. Execute:
```
python test_classifiers.py
```

## Finetuning Process

### Vision Transformer (ViT) on Deepfake FFHQ Dataset
Run the Jupyter Notebook `finetune.ipynb` to perform finetuning on the ViT model using the Deepfake FFHQ dataset (or possibly another new dataset).

Follow these instructions to ensure the correct setup and execution of scripts within the project.

