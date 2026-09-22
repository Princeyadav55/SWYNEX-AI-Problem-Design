\# AI Spam Message Classification



\## 1. Problem Statement



Unwanted and spam messages are common in SMS and messaging systems. Manually identifying spam messages can be time-consuming and difficult when the number of messages becomes large.



The goal of this project is to build a machine learning system that can automatically classify a text message as either:



\- Spam

\- Not Spam (Ham)



The system should accept a text message from the user and predict its category using a trained machine learning model.



\---



\## 2. Target User



The target users are:



\- Individuals who want to identify suspicious messages.

\- Developers who want a simple example of text classification.

\- Organizations that need a basic automated spam filtering system.



This project is designed as a small-scale AI prototype and is not intended to replace production-level security or messaging filters.



\---



\## 3. Dataset



The project uses the SMS Spam Collection dataset.



The dataset contains labeled SMS messages with two classes:



\- ham — normal message

\- spam — unwanted or spam message



The dataset contains 5,572 messages used for this project.



The dataset is stored locally in:



data/SMSSpamCollection



\---



\## 4. Data Preparation



The dataset is loaded using pandas.



The message text is used as the input feature and the label is used as the target.



The dataset is divided into:



\- 80% training data

\- 20% testing data



A fixed random state is used so that the experiment can be reproduced.



\---



\## 5. AI Approach



The project uses a text classification pipeline consisting of:



1\. TF-IDF Vectorization

2\. Multinomial Naive Bayes Classification



\### TF-IDF



TF-IDF converts text messages into numerical features that can be processed by a machine learning algorithm.



\### Multinomial Naive Bayes



Multinomial Naive Bayes is used to classify the numerical text features into spam or ham categories.



The complete pipeline is trained using the training dataset.



\---



\## 6. System Workflow



The system works in the following steps:



User enters a message.



↓



The message is passed to the trained machine learning pipeline.



↓



TF-IDF converts the message into numerical features.



↓



The Naive Bayes classifier predicts the category.



↓



The system displays:



SPAM



or



NOT SPAM



\---



\## 7. Success Criteria



The model is evaluated using:



\- Accuracy

\- Precision

\- Recall

\- F1-score

\- Confusion Matrix



The evaluation focuses on both overall performance and the model's ability to correctly identify spam messages.



The initial experiment achieved:



Accuracy: 96.05%



Spam Precision: 1.00



Spam Recall: 0.70



Spam F1-score: 0.83



Ham Precision: 0.96



Ham Recall: 1.00



Ham F1-score: 0.98



These results are based on the held-out test set of 1,115 messages.



\---



\## 8. Constraints and Limitations



The project has several limitations:



\- The dataset is relatively small compared with real-world messaging systems.

\- The model is trained on the SMS Spam Collection dataset and may not generalize perfectly to every type of modern spam message.

\- Spam patterns can change over time.

\- The model may make incorrect predictions for messages with unusual wording.

\- The current system is a prototype and does not include advanced production-level security features.



\---



\## 9. Expected Outcome



The expected outcome is a working AI prototype that allows a user to enter a message and receive a prediction of whether the message is spam or not spam.



The trained model is saved as:



model/spam\_classifier.pkl



A Streamlit-based web interface is provided through:



app.py



\---



\## 10. Project Objective



The main objective is to demonstrate how a practical AI problem can be converted into a machine learning solution using:



\- Real-world text data

\- Data preparation

\- Feature extraction

\- Machine learning

\- Model evaluation

\- A simple user interface



The project also demonstrates the complete workflow from problem definition to a working AI prototype.

