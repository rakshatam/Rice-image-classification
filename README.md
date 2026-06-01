# Rice Variety Classification using ConvNeXtV2NanoCBAM

## 🌾 Project Overview
Accurate classification of rice varieties is a critical task in agriculture, food science, and commerce. This project addresses the urgent need for an automated, scalable, and highly accurate system to distinguish between different rice varieties based on visual characteristics. 

We propose a custom deep learning model named **ConvNeXtV2NanoCBAM**, which combines modern ConvNeXtV2 blocks with Convolutional Block Attention Modules (CBAM) to achieve high accuracy and robustness.

## ✨ Key Features
* **Custom Architecture**: Built upon ConvNeXtV2 using a computationally efficient "Nano" configuration (depths=[2,2,6,2], dims=[48,96,192,384]).
* **Attention Mechanism**: Integrates CBAM (Channel and Spatial Attention) after each main stage to sequentially refine feature maps by learning "what" and "where" to focus.
* **Robust Training**: Utilizes strong data augmentation (RandomRotation, RandomAffine, ColorJitter) and advanced regularization (DropPath rate 0.2, Weight Decay 5e-5).
* **Advanced Learning Rate Schedule**: Implements a SequentialLR scheduler with a 5-epoch linear warmup followed by Cosine Annealing to ensure stable convergence.
* **Explainable AI (XAI)**: Includes a custom batched Score-CAM implementation to visualize and explain model predictions by capturing feature maps from the final stage.

## 📊 Dataset
The model is trained and evaluated on the **Rice_Image_Dataset**, which contains 5 distinct rice classes:
1. Arborio
2. Basmati
3. Ipsala
4. Jasmine
5. Karacadag

**Data Split:**
* Training Set: 72%
* Validation Set: 18%
* Test Set: 10%

## 🚀 Performance & Results
The model achieved an outstanding test accuracy of **99.75%** (7481 correct predictions out of 7500 test images).

* **Perfect Classification**: The *Ipsala* variety was classified with 100% accuracy.
* **Minor Confusions**: Only a few misclassifications occurred, primarily showing subtle visual similarities between (Jasmine, Basmati) and (Arborio, Karacadag).

## 🧠 Explainability (Score-CAM)
To ensure the model learns relevant features rather than spurious background artifacts, we apply **Score-CAM**. The generated heatmaps confirm that the model successfully localizes the rice grains and bases its predictions on learned visual features.

## ⚙️ Hyperparameters
* `IMG_SIZE`: 224
* `BATCH_SIZE`: 128
* `EPOCHS`: 30
* `WARMUP_EPOCHS`: 5
* `LEARNING_RATE`: 3e-4
* `WEIGHT_DECAY`: 5e-5
* `OPTIMIZER`: AdamW

## 📚 References
1. Sanghyun Woo et al. *ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders*. [arXiv:2301.00808](https://arxiv.org/abs/2301.00808)
2. Sanghyun Woo et al. *CBAM: Convolutional Block Attention Module*. [ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf)
3. Wang, H. et al. *Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks*. [CVPR Workshops 2020](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w1/Wang_Score-CAM_Score-Weighted_Visual_Explanations_for_Convolutional_Neural_Networks_CVPRW_2020_paper.pdf)
4. Avuçlu, E., & Köklü, M. *Rice Image Dataset*. [Dataset](https://www.muratkoklu.com/datasets/)
5. Shorten, C., & Khosh goftaar, T. M. (2019). A survey on Image Data Augmentation for Deep Learning. *Journal of Big Data*. [Link](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0)
