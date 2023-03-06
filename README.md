# Wordle-DataMining
<<<<<<< HEAD
We solved the problem C of 2023 MCM contest.
=======

## Tasks

We solved the problem C of 2023 MCM contest, this is a data mining program with the following tasks:

- The number of reported results vary daily. Develop a model to explain this variation and  use your model to create a prediction interval for the number of reported results on March  1, 2023. Do any attributes of the word affect the percentage of scores reported that were  played in Hard Mode? If so, how? If not, why not? 
- For a given future solution word on a future date, develop a model that allows you to  predict the distribution of the reported results. In other words, to predict the associated  percentages of (1, 2, 3, 4, 5, 6, X) for a future date. What uncertainties are associated with  your model and predictions? Give a specific example of your prediction for the word  EERIE on March 1, 2023. How confident are you in your model’s prediction?
- Develop and summarize a model to classify solution words by difficulty. Identify the  attributes of a given word that are associated with each classification. Using your model,  how difficult is the word EERIE? Discuss the accuracy of your classification model. 
- List and describe some other interesting features of this data set.

## Our work

Wordle is an addictive and challenging word puzzle game that has taken the world by storm. Our team conducted data mining on the Wordle dataset. 

In this study, we proposed two methods for predicting the number of reported results time series: an ARIMA model and a Long-Short Term Memory network. We compared the results of both models and found that the LSTM network performed significantly better. Combining the results from both models, we determined that the prediction interval is from 12320 to 17014, with a most likely value of 14667. Next, we defined several attributes of the target word, including the Posterior Position Probability, and calculated the Pearson correlation between each attribute and the percentage of hard mode. However, we found that none of the attributes showed a significant correlation with the percentage of hard mode.

We then used an Extreme Gradient Boosting model to forecast the distribution of reported results. Since we hypothesized that the distribution followed a Skewed Distribution, we performed a Kolmogorov-Smirnov test, which confirmed our hypothesis that the distribution is not a Gaussian Distribution. Additionally, we conducted a Principal Component Analysis on the attributes of the word and utilized autocorrelation test finding no correlation with time and the distribution. After fine-tuning the parameters of the model, we were able to predict the mean, variance, and skewness to describe the distribution. We evaluated the stability of the model through five-fold cross-validation. The evaluation parameters obtained for each subset are almost consistent, which indicates that the model is very stable, and we have confidence in this model. We elaborated on the model's uncertainty from three aspects: sampling uncertainty, data uncertainty and interpretational uncertainty.

Finally, we attempted to classify solution words by difficulty and discover other interesting features. We applied the unsupervised clustering algorithm DBSCAN to the distribution of the number of attempts required to guess a word and identified three categories of words: easy, moderate, and hard, based on the mean and variance of the distribution. We then used the labels from the clustering and data to train a Multi-Layer Perceptron Classifier and the accuracy is 98.61\%. As an example, the word ``EERIE" was classified as being in the easy class. About other interesting features of this data set, we discovered the rate of choosing hard mode tended to be stable as time goes by.

## Date Set Description

Data File: Problem C Data Wordle.xlsx and Data File Entry Descriptions:

- Date: The date in mm-dd-yyyy (month-day-year) format of a given Wordle puzzle. 
- Contest number: An index of the Wordle puzzles, beginning with 202 on January 7, 2022. 
- Word: The solution word players are trying to guess on the associated date and contest  number. 
- Number of reported results: The total number scores that were recorded on Twitter that  day. 
- Number in hard mode: The number of scores on Hard mode recorded on Twitter that day. 
- 1 try: The percentage of players solving the puzzle in one guess. 
- 2 tries: The percentage of players solving the puzzle in two guesses. 
- 3 tries: The percentage of players solving the puzzle in three guesses. 
- 4 tries: The percentage of players solving the puzzle in four guesses. 
- 5 tries: The percentage of players solving the puzzle in five guesses. 
- 6 tries: The percentage of players solving the puzzle in six guesses. 
- 7 or more tries (X): The percentage of players that could not solve the puzzle in six or fewer  tries. Note: the percentages may not always sum to 100% due to roundin
>>>>>>> 9e564ea (first commit)
