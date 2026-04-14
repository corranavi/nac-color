# Deep Learning Framework for Breast Cancer Response Assessment (Master Thesis)

This repository contains the code developed for my Master Thesis in Data Science and Engineering at Politecnico di Torino (July 2024). The Master Thesis, not included in this repo, is available for private sharing upon request.

The project focuses on developing a deep learning framework to support clinicians in assessing **breast cancer regression after Neoadjuvant Chemotherapy (NAC)** using pre- and post-treatment MRI scans.

The proposed approach aims to improve classification performance under **extremely limited data availability** by leveraging **transfer learning** and carefully designed architectural strategies based on pre-trained CNNs (ResNet backbones).


---

## Problem Overview

The objective is to classify treatment response from MRI scans by analyzing changes between pre- and post-NAC imaging. This setting is particularly challenging due to:

- Limited labeled medical imaging data  
- High variability in MRI acquisition  
- Need for robust generalization in a clinical context  

To address these challenges, the framework combines:
- **Multi-branch deep learning architectures** to process complementary imaging inputs  
- **Automated colorization modules** to enhance representational capacity and improve transfer learning from pre-trained models  
- **Fine-tuning of ResNet-based architectures** on small datasets  

---

## Methodology

The framework is built around a transfer learning strategy using pre-trained CNNs (ResNet variants), adapted through:

- Multi-channel / multi-branch input design  
- Architectural customization for medical imaging fusion  
- Fine-tuning strategies tailored to small datasets  
- Experiment tracking across multiple learning stages  

The experimental setup is designed to systematically evaluate different architectural choices and learning configurations.

---

## Code Structure

- `train.py`  
  Main training pipeline, including model initialization, training loop, and experiment orchestration.

- `evaluate.py`  
  Evaluation script for testing trained models on held-out data.

- `dataset_lib.py`  
  Data loading, preprocessing, and dataset construction utilities for MRI inputs.

- `visualize_and_predicit.py`  
  Inference and visualization utilities for qualitative model assessment.

- `model.py`  
  PyTorch Lightning model wrapper implementing training logic and custom components.

- `architectures_monobranch.py` / `architectures_multibranch.py`  
  ResNet-based architectures for single- and multi-branch experimental settings.

- `utils/`  
  Utility modules supporting training, logging, and evaluation workflows.

---

## Data Privacy

This project was developed on a **private and sensitive medical dataset**.  
As a result, the code cannot be executed or fully reproduced without access to the original data.

---

## Execution

Training can be launched using the following command (general prompt):

```bash
python train.py \
  --epochs=$EPOCHS \
  --wanb_project_name=$WANB_PROJECT_NAME \
  --batch=$BATCH_SIZE \
  --folds=$FOLDS \
  --input_path=$INPUT \
  --class_weight=1 \
  --exp_name=$EXP_NAME \
  --architecture=$ARCHITECTURE \
  --learning_rate=$LR \
  --l2_reg=$WD
```


### Notes:

* Experiments are organized using `exp_name` (multi-stage learning protocol)
* Model variants are selected via `architecture`
* Training is tracked using Weights & Biases

---

## Key Contributions

* Design of a **deep learning framework for medical image-based treatment assessment**
* Integration of **transfer learning under strong data scarcity constraints**
* Development of **multi-branch CNN architectures for multimodal MRI representation**
* Introduction of **automated colorization to improve pre-trained model adaptability**
* Systematic experimental evaluation of architectural and training strategies

---

## Objective

The goal of this work is to explore how transfer learning and architectural design can be combined to build robust medical imaging models under limited data conditions, with the broader aim of supporting clinical decision-making in oncology.
