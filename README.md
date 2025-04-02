# ML Project - Student Performance Predictor

This project aims to predict student performance based on various factors such as Gender, Ethnicity, Parental Level of Education, Lunch, and Test Preparation Course. The goal is to build a robust Machine Learning model using Python to predict student scores.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Allowed Values for New Data](#allowed-values-for-new-data)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project builds a Student Performance Predictor using Machine Learning techniques. It includes steps such as data ingestion, data transformation, model training, and prediction using the pipelines.

## Dataset
- **Source:** [Kaggle - Student Performance Dataset](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977)
- **Size:** 8 columns, 1000 rows
- **Features:** Gender, Ethnicity, Parental Level of Education, Lunch, Test Preparation Course, and Test Scores

## Installation
To get started, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/sujeetgund/mlproject-udemy.git
cd mlproject-udemy
```

2. Create a virtual environment and activate it:
```bash
python -m venv env
source env/bin/activate # For Linux/macOS
env\Scripts\activate # For Windows
```

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

4. Install the project using:
```bash
pip install -e .
```
After installation, a folder named `ml_project_udemy.egg-info` will be created.

## Project Structure
```
mlproject-udemy/
│
├── README.md
├── setup.py
├── requirements.txt
├── logs/
│   └── *.txt # Log files
├── artifacts/
│   ├── raw_data.csv
│   ├── train.csv
│   ├── test.csv
│   ├── preprocessor.pkl # Saved preprocessor after transformation
│   ├── model.pkl # Trained model file
├── notebooks/
│   ├── eda.ipynb
│   ├── model_training.ipynb
│   └── data/
│       └── stud.csv
├── ml_project_udemy.egg-info/
└── src/
    ├── __init__.py
    ├── logger.py
    ├── exception.py
    ├── utils.py
    ├── components/
    │   ├── __init__.py
    │   ├── data_ingestion.py
    │   ├── data_transformation.py
    │   ├── model_trainer.py
    └── pipelines/
        ├── __init__.py
        ├── train_pipeline.py
        ├── predict_pipeline.py
```

### Description of Main Modules:
- **`logger.py`**: Handles logging for tracking events, stored in the `logs` folder.
- **`exception.py`**: Custom exception handling.
- **`utils.py`**: Utility functions for data processing.
- **`data_ingestion.py`**: Handles data loading. After running, the `artifacts` folder will contain:
  - `raw_data.csv`: The original dataset.
  - `train.csv`: Training data split.
  - `test.csv`: Testing data split.
- **`data_transformation.py`**: Prepares and transforms data for modeling. After running, it generates:
  - `preprocessor.pkl`: The saved preprocessor object.
  - Transformed train and test data arrays.
- **`model_trainer.py`**: Trains multiple machine learning models, selects the best one based on R2 score, and saves it as `model.pkl`.
- **`train_pipeline.py`**: End-to-end pipeline for training.
- **`predict_pipeline.py`**: Pipeline for making predictions. You can modify `predict_pipeline.py` to use different student data for predictions.
- **`notebooks/eda.ipynb`**: Exploratory Data Analysis notebook.
- **`notebooks/model_training.ipynb`**: Model training and evaluation notebook.
- **`notebooks/data/stud.csv`**: Student performance dataset.

## Usage
### Running the Full Project
To run the full project, execute:
```bash
python src/pipelines/train_pipeline.py
```
This will handle data ingestion, transformation, and model training.

### Running Predictions
To make predictions, execute:
```bash
python src/pipelines/predict_pipeline.py
```
If you want to predict using different student data, modify the following section inside `predict_pipeline.py`:
```python
students_data = CustomData(
    records=[
        StudentExamRecord(
            gender="male",
            race_ethnicity="group B",
            parental_level_of_education="some college",
            lunch="standard",
            test_preparation_course="none",
            reading_score=72,
            writing_score=83,
        ),
        StudentExamRecord(
            gender="female",
            race_ethnicity="group C",
            parental_level_of_education="bachelor's degree",
            lunch="free/reduced",
            test_preparation_course="completed",
            reading_score=88,
            writing_score=92,
        ),
    ]
)
```

## Allowed Values for New Data
- **Gender:** `male`, `female`
- **Race/Ethnicity:** `group A`, `group B`, `group C`, `group D`, `group E`
- **Parental Level of Education:** `some high school`, `high school`, `some college`, `associate's degree`, `bachelor's degree`, `master's degree`
- **Lunch:** `standard`, `free/reduced`
- **Test Preparation Course:** `none`, `completed`
- **Reading Score & Writing Score:** Numerical values between 0 and 100

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to your branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

