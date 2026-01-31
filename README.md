# Student Performance Indicator

## 📌 Project Overview
The **Student Performance Indicator** is a Machine Learning Web Application designed to understand how various factors—such as demographic background, parental education level, and test preparation—influence a student's academic performance. The goal is to predict students' test scores (specifically Math Score) based on these input features.

This project implements an end-to-end Machine Learning pipeline, from data ingestion and transformation to model training and deployment using **Flask**.

## 🚀 Features
- **User-Friendly Interface**: A simple web form to input student details.
- **Data Preprocessing**: Handles missing values, performs one-hot encoding for categorical variables, and standardizes numerical features.
- **Multiple Models Scored**: The system trains and evaluates multiple regression models to find the best performer:
    - Linear Regression
    - Lasso, Ridge
    - K-Neighbors Regressor
    - Decision Tree
    - Random Forest Regressor
    - XGBoost Regressor
    - CatBoosting Regressor
    - AdaBoost Regressor
- **Best Model Selection**: Automatically saves the model with the highest R2 score.
- **REST API / Web App**: Built with Flask for easy interaction.

## 🛠️ Technologies Used
- **Language**: Python 3.x
- **Web Framework**: Flask
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn, CatBoost, XGBoost
- **Visualization**: Matplotlib, Seaborn
- **Deployment**: Ready for deployment (e.g., AWS Elastic Beanstalk, Azure Web App)

## 📂 Project Structure
```
├── artifacts/          # Stores the trained model.pkl and preprocessor.pkl
├── catboost_info/      # CatBoost training logs
├── notebook/
│   ├── data/           # Dataset (StudentsPerformance.csv)
│   ├── EDA.ipynb       # Exploratory Data Analysis
│   └── ModelTraining.ipynb  # Model training experiments
├── src/                # Source code
│   ├── components/     # Data ingestion, transformation, model training modules
│   ├── pipeline/       # Prediction and training pipelines
│   ├── exception.py    # Custom exception handling
│   ├── logger.py       # Logging configuration
│   └── utils.py        # Utility functions (save/load objects)
├── templates/          # HTML templates for the Flask app
├── app.py              # Main Flask application entry point
├── requirements.txt    # List of dependencies
├── setup.py            # Package setup script
└── README.md           # Project documentation
```

## ⚙️ Installation

1. **Clone the repository** (if applicable) or download the source code.

2. **Create a virtual environment** (recommended):
   ```bash
   conda create -p venv python=3.8 -y
   conda activate venv/
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Usage

1. **Run the Flask Application**:
   ```bash
   python app.py
   ```
   The application will start on `http://127.0.0.1:5000/`.

2. **Access the Web Interface**:
   - Open your browser and go to `http://127.0.0.1:5000/`.
   - Navigate to the prediction page (usually `/predictdata` or via a button).
   - Fill in the form with student details and submit to see the predicted score.

## 📊 Dataset
The project uses the [Student Performance in Exams](https://www.kaggle.com/spscientist/students-performance-in-exams) dataset.
**Input Features:**
- `gender`: Sex of the student
- `race_ethnicity`: Ethnicity group
- `parental_level_of_education`: Parent's highest education
- `lunch`: Type of lunch (standard/free/reduced)
- `test_preparation_course`: Completed or not
- `reading_score`: Score in reading
- `writing_score`: Score in writing

**Target Variable:**
- `math_score`: Predicted Score

## 💡 Key Learnings
- Setting up a modular Machine Learning project structure.
- Handling Custom Exceptions and Logging in Python.
- Building specific Data Transformation pipelines.
- Automating Model Training and Evaluation.
- Deploying ML models using Flask.

## 👤 Author
**Sumeet** (pal.sumeetkumar@gmail.com)
