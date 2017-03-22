# Machine-Learning-in-TabPy
End to end modeling workflow to train and load a model into TabPy for use in Tableau.

This Repo incudes all required materials to train a Random Forest Classifier, a widely applicable general purpose classifier, to predict whether patients will get diabetes within 5 years given a range of diagnostics.

The data is contained in the pima-indians-diabetes.csv text file, the code for loading the data and training the model is in the Train Pima Diabetes Random Forest Classification.ipynb. This is a Jupyter Notebook intended to be run with the Jupyter package. The JupyterPimaForest.pkl contains the a preserved version of the trained model that can be loaded to TabPy. Finally, the Pima Diabetes Load to Server.ipynb contains the code to load the model to a running TabPy Server (https://github.com/tableau/TabPy).

This sample outlines a data science workflow using Tableau to visualize and deploy the results of a machine learning model.
