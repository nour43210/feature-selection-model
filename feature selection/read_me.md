Genetic Algorithm Feature Selection Web Application
Overview
This Streamlit web application implements a Genetic Algorithm (GA) for feature selection in machine learning tasks. It allows users to upload a dataset, select a target variable, configure GA parameters, and visualize the results of feature selection and model performance.
Features

Data Upload: Supports CSV and Excel file formats for dataset input.
Preprocessing: Handles both numerical and categorical data with automatic encoding and scaling.
Model Selection: Choose from Random Forest, Gradient Boosting, or SVM classifiers.
Genetic Algorithm: Configurable parameters including population size, number of generations, crossover probability, mutation probability, and tournament size.
Performance Metrics: Displays Accuracy, F1 Score, Precision, Recall, ROC AUC, and feature reduction statistics.
Visualizations: Includes:
Model performance bar plot
Confusion matrix
ROC curve (when applicable)
Feature importance plot
SHAP values summary plot
GA convergence plot


Export: Download all visualizations and results as a ZIP file containing PNG images and CSV files with selected features and metrics.

Requirements
To run the application, ensure you have the following Python packages installed:
pip install streamlit pandas numpy matplotlib seaborn scikit-learn deap shap

Usage

Run the Application:
streamlit run app.py

Replace app.py with the name of your Python script containing the code.

Upload Dataset:

Use the sidebar to upload a CSV or Excel file.
Select the target variable from the dropdown menu.
Click "Preprocess Data" to prepare the dataset.


Configure GA Parameters:

Choose a model (Random Forest, Gradient Boosting, or SVM).
Set population size, number of generations, crossover probability, mutation probability, and tournament size.
Click "Run Feature Selection" to execute the genetic algorithm.


View Results:

Preview the dataset and split information (training/test samples).
Explore performance metrics, selected features, and visualizations.
Download all results as a ZIP file using the "Export All Visualizations" button.



Code Structure

Class GeneticFeatureSelector:
Manages data loading, preprocessing, GA-based feature selection, model training, and result generation.
Methods include load_data, preprocess_data, run_genetic_algorithm, get_performance_metrics, and explain_model.


Visualization Functions:
plot_confusion_matrix: Generates a confusion matrix heatmap.
plot_roc_curve: Plots the Receiver Operating Characteristic curve.
get_image_download_link: Creates downloadable links for plots.


Main Function:
Sets up the Streamlit interface with sidebar inputs and result displays.



Notes

The application uses the deap library for genetic algorithm implementation and shap for model interpretability.
Categorical variables are automatically encoded using one-hot encoding, and numerical features are scaled using StandardScaler.
The GA optimizes a multi-objective fitness function based on accuracy, F1 score, precision, and recall.
Visualizations are generated using matplotlib and seaborn, with download options for high-resolution PNGs.
The application is styled with custom CSS for improved user experience.

Limitations

Requires a dataset with a clear target variable and sufficient features for meaningful selection.
SHAP explanations may not be available for all models or may fail for complex datasets.
Performance depends on the dataset size and complexity, as well as the chosen model's computational requirements.

Future Improvements

Add support for additional machine learning models.
Implement cross-validation for more robust performance metrics.
Enhance visualization interactivity with tools like Plotly.
Optimize GA performance for larger datasets.

License
This project is licensed under the MIT License.