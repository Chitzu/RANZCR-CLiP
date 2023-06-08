# RANZCR-CLiP Project


## Project Overview 🏥
In this project I implemented machine learning algorithms to solve a medical imaging problem based on a Kaggle Competition - RANZCR CLiP - Catheter and Line Position Challenge.

This project is based on the Kaggle Competition "RANZCR CLiP" (https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-classification/) and the main idea is based on the Classification of the presence and correct placement of tubes on chest x-rays to save lives

## About 

I developed this project with Pytorch in which I train a model with a CNN (ResNet18) on the RANZCR CLiP dataset,after processing the data, test the model and save each loss per epoch, the best performance and the model itself. You can configure the data such as the loss function, batch size, test and train size, but also where to save the experiment data with a json file.

