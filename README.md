# Hospital Helper 🏥

Hospital Helper is a machine learning project that uses hospital data to perform three tasks:

- **Classification:** Predicting cardiovascular disease.
- **Clustering:** Grouping similar cardiovascular data.
- **Regression:** Predicting hospital Length of Stay (LOS).

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

## Project Structure

- `app.py`: Main Streamlit application.
- `Data/`: Contains the datasets.
- `Models/`: Contains classification, clustering, and regression models.
- `preprocessing/`: Contains data preprocessing files.
- `Save_Models/`: Contains the saved models and results.
- `Plots/`: Contains generated plots.
- `src/`: Contains the Jupyter notebook.

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```
