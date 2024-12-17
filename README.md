# AI to detect hate in tweets

## Description
This project implements machine learning models to analyze tweets and classify them into three categories: "Hate Speech", "Offensive Speech", or "Accepted Speech". The dataset used is the "Hate Speech and Offensive Language" dataset from Kaggle. To analyse different features of the tweet, we utilized 3 different machine learning algorithms:

- TF-IDF with Multinomial Naive Bayes: Identifies offensive words in tweets.
- LSTM-based Deep Neural Network: Analyzes the meaning of words in sequential data.
- Perspective API: Analyzes sentiment using a convolutional neural network, returning values such as toxicity or profanity.

These models are trained on a dataset of labeled tweets, that has being first preprocesed, removing unecessary characters suchs as emojis or extra symbols. 

## Table of Contents
* [Tools](#tools)
* [Installation](#installation)
* [Usage](#usage)
* [Credits](#credits)
* [License](#license)
* [Questions](#questions)

## Tools
- Python
- Visual Studio Code
- GitHub
- Kaggle
- Perspective API 

## Installation
To use this code you need to use a python version 3.9-3.12. After downloading the code you will have to pip install the next dependencies in order to run the code.
- nltk
- bs4
- pandas
- scikit-learn
- SMOTE
- pickle
- matplotlib
- numpy
- tensorflow
- keras
- googleapiclient

Additionally, you will need a Perspective API key, which you can obtain from [Perspective API](https://www.perspectiveapi.com/). Once you have the key, open the file perspectiveAPI_model.py and paste it into the variable API_KEY

## Usage

You can skip Step 1 if the preprocessed_hate_speech_dataset.py file already exists. However, if you replace the dataset, you must also remove any files ending with .pkl, and then run all the steps again: 

1. Run the file preprocessed_data.py. This will preprocess the data in the labeled_data.csv dataset and create a new dataset (preprocessed_hate_speech_dataset.csv) ready for training the machine learning models.

2. Run the file ml_naive_bayes_model.py. This will create the Naive Bayes model and save it as a .pkl file. The model will then be trained using the preprocessed dataset. After training, the program will display a graph showing the distribution of correct predictions. 

3. Stop the terminal by entering quit, and then run the file deep_learning_LSTM_model.py. The LSTM model will begin training. Once training is complete, you can enter your own input for classification.

4. Stop the terminal by entering quit, and then run the file perspectiveAPI_model.py. This will initiate an API request to analyze the toxicity sentiment of user inputs. Enter quit to stop the process once you are done.



## Credits
This code is based on source code from <a href="https://github.com/benkimmn/content-moderator">this repository</a>, which has been modified and improved for the creation of the Multinomial Naive Bayes model. Additionally, I want to express my gratitude to my friends for their invaluable support during this project. 

Created by <a href="https://github.com/nowinoa">Ainhoa Prada</a>.
        
## Questions
For any questions or issues feel free to contact me at: ap2170u@gre.ac.uk

To explore more about my projects visit my profile:

<a href="https://github.com/nowinoa">:computer:</a>

© 2024 Ainhoa Prada. Confidential and Proprietary. All Rights Reserved.
