# hotel_review

## Table of contents

    - Description
    - Technologies
    - Setup

## Description

In this project, given a set of reviews, we want to predict the rating value for each of them.

Dataset source: Kaggle  (https://www.kaggle.com/andrewmvd/trip-advisor-hotel-reviews)

## Technologies

   - Python (Jupyter Notebook)
   - nltk
   - matplotlib
   - numpy 
   - pandas 
   - sklearn

## Tasks performed
   - Data exploration
   - Data cleaning 
   - Testing of different classification algorithms (Logistic Regression, SVM, Decision Tree, Multinomial Naive Bayes, Neural Networks)

## Results

 Model                | Training score  | Test Score   | ValidationScore
----------------------| --------------  | ------------ |----------------
LogisticRegression    |   0.764438      |0.609213      |   0.611906
LinearSVC             |   0.854645      |0.612922      |   0.611906
DecisionTreeClassifier|   0.696437      |0.492875      |   0.477228
MultinomialNB         |   0.791931      |0.583642      |   0.576773
OneVsRestClassifier   |   0.997641      |0.521374      |   0.522121

Logistic regression and SVM with linear kernel were the first choices because they are simple and work similarly for problems with a similar number of features and training data. Given the overfitting problem, after using various techniques, we proceeded to test other models such as decision trees and Nayve Bayes, but without getting better results. Because the size of the dataset is very small, neural networks had not been considered previously, but as a last attempt we tried using them and working on the number of hidden layers. The results worsen significantly by increasing even slightly the number of hidden layers or trying to work on the regularization parameter. In general, the various models used fail to make satisfactory predictions, all suffering from overfitting. The decidedly large gap present in the learning curves is further confirmation of this. Several techniques have been applied to improve the models: the use of training set division, validation set, test set, increasing the regularization parameter or decreasing the C parameter, and using fewer features. All of the solutions adopted in many cases proved ineffective, at best very good accuracy was obtained on the training set, but very bad for the validation and test sets, an indication that the model has strong generalization difficulties when it comes to making predictions on new data. Some improvement was found only with SVM, with slightly better accuracy on all three sets.

