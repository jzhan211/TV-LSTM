# TV-LSTM
This repository contains the PyTorch implementation for:

**Multimodal Deep Learning for Predicting the Progression of Late Age-related Macular Degeneration Using Longitudinal Fundus Images and Genetic Data**

**Abstract**: Age-related macular degeneration (AMD) is the leading cause of blindness in developed countries. Predicting its progression is crucial for preventing late-stage AMD, as it is an irreversible retinal disease. Both genetic factors and retinal images are instrumental in diagnosing and predicting AMD progression. Previous studies have explored automated diagnosis using single fundus images and genetic variants, but they often fail to utilize the valuable longitudinal data from multiple visits. Longitudinal retinal images offer a dynamic view of disease progression, yet standard Long Short-Term Memory (LSTM) models assume consistent time intervals between visits, limiting their effectiveness in real-world settings. To address this limitation, we propose Time-Varied Long Short-Term Memory (TV-LSTM), which accommodates irregular time intervals in longitudinal data. Our approach integrates longitudinal fundus images and AMD-associated genetic variants to predict late AMD progression. TV-LSTM achieved an AUC-ROC of 0.9479 and an AUC-PR of 0.8591 for predicting late AMD within two years, using data from four visits with varying time intervals. 

## Installation
Try `pip install -r requirments.txt` to install required packages.
**Dependencies**:
   - Python 3.7
   - PyTorch + Torchvision
   - Pytorch Lightning (for data preparation)
   - Tensorflow
   - Scikit-learn + Scikit-image
   - PIL

## Code Overview
   - [test_lateAMDin2_tvlstm_4v.py](./test_lateAMDin2_tvlstm_4v.py) : for prediction of late AMD in two years using 4 history fundus images and genotypes

     `python3 ./test_lateAMDin2_tvlstm_4v.py --image_folder ./test_data/Available_Fundus/ --history_visit_num 4 --input_dim 1077 --geno_file ./test_data/test_subject_52SNPs.txt --weights_file ./TVLSTM_g_4v_weights.pth`

## Trained Checkpoint Models
   - [TVLSTM_g_4v_weights.pth](./TVLSTM_g_4v_weights.pth) : TV-LSTM with 4 history fundus and genotypes

## Contact 
If you have any questions, please feel free to contact us through email jipeng.zhang@pitt.edu).
