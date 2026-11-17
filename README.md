# Machine Learning Notebooks: Intro to ML, Deep Learning & AI (Rutgers–Newark, Spring 2026)

Google Colab notebooks from my machine learning course, written in Python. Each notebook keeps its saved outputs (tables, charts, metrics), so you can read the results on GitHub without running anything.

| Notebook | What it does | Tools |
|---|---|---|
| [Titanic_Model_Comparison.ipynb](Titanic_Model_Comparison.ipynb) | Predicts Titanic passenger survival and compares **5 models**: Decision Tree, Logistic Regression, SVM, Random Forest and KNN. Cleans the data (drops unused columns, encodes sex, fills missing ages), then evaluates each model with accuracy, precision, recall, confusion matrix, ROC AUC and 5-fold cross-validation. Random Forest scored best (≈83% test accuracy, ≈80% cross-validated). | pandas, scikit-learn, seaborn, matplotlib |
| [Wine_Classification_SVM.ipynb](Wine_Classification_SVM.ipynb) | Trains a linear Support Vector Machine to classify wines into 3 classes from chemical features, with train/test splitting and feature scaling. Reaches **82% test accuracy**, analyzed with a confusion matrix. | pandas, scikit-learn, matplotlib |
| [Data_Preprocessing_Basics.py](Data_Preprocessing_Basics.py) | First assignment: binarizing, normalizing (L1/L2), computing variance and label-encoding a small NumPy matrix. | NumPy, scikit-learn |
| [Loan_Intent_Classification.ipynb](Loan_Intent_Classification.ipynb) | Predicts a borrower's loan purpose from gender, age, education and income using a decision tree, with label encoding and stratified k-fold cross-validation. | pandas, scikit-learn |
| [Face_Detection.ipynb](Face_Detection.ipynb) | Detects faces and eyes in photos with OpenCV Haar cascades, then tunes `scaleFactor` and `minNeighbors` to cut false detections in a crowded group photo. | OpenCV, matplotlib |
| [Feature_Matching.ipynb](Feature_Matching.ipynb) | Finds a product in a cluttered shelf image by matching features: ORB with brute-force matching, SIFT with a ratio test, and FLANN-based matching. | OpenCV |
| [Apple_Stock_Price_Prediction.ipynb](Apple_Stock_Price_Prediction.ipynb) | Builds sliding 10-day windows of Apple opening prices and trains an **LSTM** (a recurrent neural network) to predict the next day's price, then plots predictions against real prices. | TensorFlow/Keras, pandas, scikit-learn |
| [Cancer_Image_CNN.ipynb](Cancer_Image_CNN.ipynb) | *(Work in progress.)* Starter convolutional neural network that classifies medical images (JPEG/PNG and DICOM scans) as cancer or not cancer. | TensorFlow/Keras, OpenCV, pydicom |

**Note:** The datasets and images these notebooks load (`Titanic.csv`, `loan_data.csv`, the photos) were uploaded to Colab at runtime and aren't included here.
