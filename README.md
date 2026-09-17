# CAP 3764 Fall 2026 - Team 1

## Project Topic
Student Performance Analysis

## Project Goal
The goal of this project is to analyze factors that may be related to student academic performance. We will use numerical and categorical variables from the Student Performance dataset to explore patterns in student characteristics, study habits, and academic outcomes.

## Dataset
We are using the UCI Student Performance dataset.

The dataset contains information about students and their academic performance, including demographic, social, school-related, and academic variables.

- Observations: 649 students
- Variables: 33
- Numerical variables: 16
- Categorical variables: 17
- Target variable: G3 (final grade)
- G3 range: 0–20

Dataset source: UCI Machine Learning Repository

## Data Collection and Cleaning
The dataset was loaded using a custom Python module located in `my_modules/data_collection_module.py`.

Initial data cleaning included:
- Checking dataset dimensions and data types
- Checking for missing values
- Checking for duplicate rows
- Removing unnecessary whitespace from column names
- Removing duplicate rows

The original dataset contained 649 rows and 33 columns. No missing values or duplicate rows were found.

## Exploratory Data Analysis
The initial EDA will include:
- Summary statistics for numerical variables
- Analysis of categorical variables
- Summary tables
- Relevant data visualizations

## Repository Structure

- `my_modules/` - Custom Python modules used in the project
- `notebooks/` - Jupyter notebooks for data collection, cleaning, and analysis
- `data/` - Local dataset files (excluded from Git tracking)
- `outputs/` - Generated project outputs
- `environment.yml` - Conda environment configuration
- `.gitignore` - Files and directories excluded from Git tracking

## Team Members
- Dariya Damirbekova
- Aiana Albert
- Mehruza Nosirova 
- Bakai Assylbekov 

## Next Steps
The next stage of the project is to perform exploratory data analysis, compare numerical and categorical variables, create visualizations, and identify preliminary patterns related to student performance.