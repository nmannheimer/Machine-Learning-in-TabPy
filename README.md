# Machine-Learning-in-TabPy
End to end modeling workflow to train and load machine learning models into TabPy for use in Tableau.

This Repo incudes all required materials to train a Random Forest Classifier, a widely applicable general purpose classifier, to predict whether patients will get diabetes within 5 years given a range of diagnostics. Also included is a Neural Net (Multi-Layer Perceptron) Regression model designed to predict the price of a home in Boston, based on a range of factors.

The data for the Random Forest model is the pima-indians-diabetes.csv text file, the code for loading the data and training the model is in the Train Pima Diabetes Random Forest Classification.ipynb. This is a Jupyter Notebook intended to be run with the Jupyter package. The JupyterPimaForest.pkl contains the a preserved version of the trained model that can be loaded to TabPy. Finally, the Pima Diabetes Load to Server.ipynb contains the code to load the model to a running TabPy Server (https://github.com/tableau/TabPy).

The data for the Neural Net is the Boston_Housing.csv text file. The code to train and evaluate the model, as well as define it as a function and deploy it to TabPy are included in the Housing Prices Model.py file.

This sample outlines data science workflows using Tableau to visualize and experiment with the results of a machine learning model.
