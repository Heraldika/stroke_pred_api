# Stroke data prediction

Data used: [Stroke prediction dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)

Project parts:

- EDA
- Stroke prediction
- BMI prediction
- Hypertension prediction
- Average glucose level prediction
- API creation with Flask

In this project I predicted stroke and various other features from the data. Then I created an **API** for these models.

The focus of the project was on trying out various ML models, using **bayesian optimization** and then trying to evaluate their performance with tools like **shap**.

To run the project yourself download the dataset, install dependencies from requirements.txt, uncomment code that creates the models and run the kernel.

To try out the api first run the `stroke.ipynb` to create the models used in it, then go to `src` directory and run the api using:

```bash
$ ./app.py
```
