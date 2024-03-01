1. Generate id_labels and facial expression labels through ./Data_Generation/Facial_label.py and ID_label.py.

2. Use ./Data_Generate/Generate_shap_values generate shap value for Facial expression tasks and identity classification tasks.

3. Use the results from step 2 in ./Data_Generate/Generate_mask.py to generate optimized masks and masked images.

Note: All generated images have the same order. All images.npy align with the labels.npy

4.Use Auditing_pytorch/run.sh to run the experiments. (Remember to replace the datasets for different experiments)
