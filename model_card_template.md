# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a Random Forest Classifier trained using scikit-learn. It predicts whether a person's annual income exceeds 50000 based on census data. The model uses 100 decision trees with a random state of 42 for reproducibility.

## Intended Use

This model is intended to classify individuals into two income brackets: greater than 50K or less than or equal to 50K per year. It is designed for educational purposes as part of a machine learning pipeline deployment project. It should not be used for making real financial or hiring decisions.

## Training Data

The model was trained on the Census Income dataset from the UCI Machine Learning Repository. The dataset contains 32561 rows and 14 features including age, workclass, education, marital status, occupation, relationship, race, sex, capital gain, capital loss, hours per week, and native country. 80 percent of the data was used for training and 20 percent for testing using a random train-test split with random state 42. Categorical features were encoded using a OneHotEncoder and the target label was binarized using a LabelBinarizer.

## Evaluation Data

The model was evaluated on the remaining 20 percent of the Census Income dataset held out during the train-test split. The same preprocessing pipeline fitted on the training data was applied to the evaluation data.

## Metrics

The model was evaluated using precision, recall, and F1 score. Precision: 0.7419, meaning 74.19 percent of predicted high earners actually earn more than 50K. Recall: 0.6384, meaning the model correctly identified 63.84 percent of actual high earners. F1 Score: 0.6863, the harmonic mean of precision and recall.

## Ethical Considerations

The Census Income dataset contains sensitive demographic features such as race, sex, and native country. The model may reflect historical biases present in the data. Performance varies across demographic slices as shown in slice_output.txt. This model should not be used to make decisions that affect individuals in employment, lending, or any other high-stakes context.

## Caveats and Recommendations

The model was trained on 1994 US Census data and may not reflect current income distributions. Performance on underrepresented groups may be unreliable. It is recommended to retrain the model on more recent data if used in any real-world application. Hyperparameter tuning was not performed and further improvements may be possible.
