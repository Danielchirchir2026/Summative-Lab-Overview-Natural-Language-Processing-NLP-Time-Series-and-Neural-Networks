# Amazon Customer Reviews Analysis

## Project Overview

This summative lab applies three core data science techniques to the
`amazon_reviews_lab.csv` dataset:

1. Natural Language Processing
2. Time-Series Analysis and Forecasting
3. Neural Network Classification

The notebook follows the assignment's `# TODO:` structure and includes
additional validation and modelling

## Dataset

The dataset contains 997 Amazon customer reviews and includes:

- Product identifiers
- Ratings from 1 to 5
- Review text and summaries
- Review timestamps
- Sentiment labels
- Review-length information

## Part 1: Natural Language Processing

The NLP section cleans and analyzes customer-review text by:

- Removing missing and duplicate records
- Converting text to lowercase
- Removing URLs, HTML, punctuation, numbers, and stopwords
- Tokenizing and lemmatizing review text
- Calculating common word frequencies
- Identifying common bigrams and trigrams
- Comparing frequent terms across sentiment categories
- Visualizing important language patterns

This analysis identifies recurring product features, customer concerns,
and themes associated with positive and negative feedback.

## Part 2: Time-Series Analysis

The time-series section examines monthly review activity and average customer
ratings by:

- Converting review timestamps to datetime format
- Calculating monthly review counts and average ratings
- Plotting review-volume and rating trends
- Calculating a three-month rolling average
- Testing stationarity using the Augmented Dickey-Fuller test
- Applying log transformation and first differencing
- Performing seasonal decomposition
- Examining ACF and PACF plots
- Comparing multiple ARIMA models using AIC, BIC, MAE, and RMSE
- Generating a 12-month forecast with a 95% prediction interval

The analysis shows that review activity was sparse in the early years but
became substantially more frequent in later years. Forecasts are interpreted
as approximate expected review volumes rather than exact monthly predictions.

## Part 3: Neural Network Classification

The neural-network section predicts customer ratings from review text by:

- Encoding ratings 1 to 5 as classes 0 to 4
- Creating stratified training, validation, and test datasets
- Converting review text into TF-IDF features
- Applying class weights to address rating imbalance
- Building a multilayer TensorFlow/Keras neural network
- Comparing ReLU and tanh activation functions
- Using dropout, batch normalization, early stopping, and learning-rate reduction
- Evaluating the selected model using accuracy, classification reports,
  confusion matrices, and a majority-class baseline

The selected tanh model achieved 58% test accuracy, which matched the
majority-class baseline. It predicted Rating 5 for every test review and failed
to classify Ratings 1 to 4 effectively. The model is therefore unsuitable for
operational five-class rating prediction in its current form.

## Key Conclusion

The NLP and time-series analyses provide useful insights for identifying
customer themes and monitoring changes in review activity. However, the neural
network requires further improvement because its results are dominated by the
majority Rating 5 class.

Recommended improvements include:

- Increasing minority-class representation
- Testing oversampling methods
- Tuning class weights and network architecture
- Comparing alternative classification algorithms
- Using macro F1-score for model selection
- Considering broader negative, neutral, and positive rating categories

## Technologies Used

- Python
- Jupyter Notebook
- pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- scikit-learn
- statsmodels
- TensorFlow/Keras

## Running the Project

1. Place `amazon_reviews_lab.csv` in the same folder as the notebook.
2. Install the required Python libraries.
3. Open the notebook in Jupyter Notebook or VS Code.
4. Select the correct Python environment.
5. Restart the kernel and run all cells from top to bottom.
6. Confirm that all outputs, charts, and reflections are complete.

## Limitations

- Early review activity is sparse.
- Rating classes are significantly imbalanced.
- TF-IDF does not fully capture context, sarcasm, or word order.
- The neural network treats ordered ratings as separate classes.
- Time-series forecasts cannot anticipate external events absent from the data.

## Author

**Daniel Chirchir**
  
