Code for Master Thesis in Data Science and Engineering @ Politecnico di Torino, July 2024.

Disclaimer: This code runs on a private dataset (on sensitive data), therefore it is not possible to replicate it without the underlying dataset.

To run the code, the following script has to be used (general form):
python train.py --epochs=$EPOCHS  --wanb_project_name=$wanb_project_name  --batch=$batchsize --folds=$FOLDS --input_path=$INPUT --class_weight=1 --exp_name=$exp_name --architecture=$architecture --learning_rate=$LR --l2_reg=$WD

Note that the experiments have been based on a multi-stage learning setting, relying on the experiment name ('exp_name' parameter) and on the chosen architecture ('architecture' parameter).
Feel free to contact me for further clarification.

**Summary of code structure**
- train.py : the main file for training the model
- evaluate.py: used to evaluate the trained model on the test dataset
- dataset_lib.py: file containing all the custom methods needed to ingest and prepare input data.
- visualize_and_predicit: used to make prediction using one of the pre-trained models and perform some visualizations.
- model.py: wrapper Lightning model with custom methods
- architectures_monobranch.py and architectures_multibranch.py: ResNet-based architectures used in the experiments.

utils: folder containing util files for supporting the several steps of the training.
