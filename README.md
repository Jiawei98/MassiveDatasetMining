# MassiveDatasetMining

## Summary

In this project, we are focusing on Fashion Product Images Dataset, which contains 44440 high-res product image files. Our purpose is to build a classifier based on CNN (Convolutional Neural Network), which can predict the category of a product by using its image. We use CHTC to improve the computational efficiency. To get an overall insight into the dataset and methodology feasibility, we show our workflow here.

![Workflow](https://github.com/ElenaXu98/MassiveDatasetMining/blob/main/figures/workflow.png)

In general, we first preprocessed the image on CHTC to get dimensional reduced, grayscale image for model training. Then, we split these images into training set (60%), validation set (30%), and test set(10%). After that, we used parallel computation on CHTC to fit multiple CNN models under LeNet5 architecture with different hyperparameters on the training set in order to find the best hyperparameter combination. The accuracy of our final model on test set is 88%.

# Files

- `\code\CNN` contains the code for CNN model on CHTC.
- `\code\build_by_packages` contains the code to build R and Python environment, and essential packages.
- `\code\convertImages\` contains the code for image preprocessing.
- `\drafts` contains the proposal, first and second version of reports.
- `\figures` contains the figures rendered in our project.
- `\final_draft` is our final report.

## Group members

Jiawei Huang (jhuang455)  
Yinqiu Xu (yxu475)  
Yike Wang (wang2557)  
Zijun Feng (zfeng66)  
Hao Jiang (hjiang266)
