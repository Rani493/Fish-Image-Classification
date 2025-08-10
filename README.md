# Fish-Image-Classification
This project successfully developed and implemented a deep learning solution for classifying images of fish into multiple categories. The core of the project was built using Python, TensorFlow, and the Keras API.

The key components and steps of the project included:

Dataset Preparation: The project utilized a multi-category fish image dataset. The images were organized into train, validation, and test directories to properly split the data for model training and evaluation. The ImageDataGenerator from Keras was used for real-time data loading and augmentation, which helped to improve the model's ability to generalize to new images.

Model Development: Two distinct deep learning models were developed:

A custom Convolutional Neural Network (CNN), built from scratch, to learn features directly from the fish images.

A transfer learning model based on the pre-trained MobileNetV2 architecture. This approach leveraged a model that had already learned to identify a wide range of features from the large ImageNet dataset, making it highly effective for this classification task.

Model Training and Saving: Both models were trained and monitored using an EarlyStopping callback to prevent overfitting. Once training was complete, the best-performing models were saved in the HDF5 format (.h5) for later use.

Web Application Deployment: A user-friendly web application was built using Streamlit. This application allows a user to upload a fish image, which is then sent to the trained model for classification. The app displays the predicted fish name, its assigned category, and the model's confidence level for the prediction.

This project demonstrates a comprehensive workflow for an image classification task, from data handling and model development to practical deployment in a web application.
