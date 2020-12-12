# MassiveDatasetMining

## Summary

In this project, we are focusing on Fashion Product Images Dataset, which contains 44440 high-res product image files. Our purpose is to build a classifier based on CNN (Convolutional Neural Network), which can predict the category of a product by using its image. We use CHTC to improve the computational efficiency. To get an overall insight into the dataset and methodology feasibility, we show our workflow here (Figure 1).

![Workflow](../figures/workflow.png){width=90%}

In general, we first preprocessed the image on CHTC to get dimensional reduced, grayscale image for model training. Then, we split these images into training set (60%), validation set (30%), and test set(10%). After that, we used parallel computation on CHTC to fit multiple CNN models under LeNet5 architecture with different hyperparameters on the training set in order to find the best hyperparameter combination. The accuracy of our final model on test set is 88%.

## Group members

Jiawei Huang (jhuang455)  
Yinqiu Xu (yxu475)  
Yike Wang (wang2557)  
Zijun Feng (zfeng66)  
Hao Jiang (hjiang266)
