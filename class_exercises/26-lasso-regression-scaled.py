import pandas as pd
import numpy as np

from sklearn.linear_model import LassoCV

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

if __name__ == "__main__":

    df = pd.read_csv("data/hitters/hitters.csv").dropna(how="any")
    y = df["Salary"].values
    X = df.drop("Salary", axis="columns").select_dtypes("number")

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model = make_pipeline(StandardScaler(), LassoCV(cv=3, alphas=np.logspace(0.00000001, 3, 100)))

    model.fit(X_train, y_train)
    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_hat_train)
    test_mse = mean_squared_error(y_test, y_hat_test)


    results_df = pd.DataFrame({
        "feature": model[0].feature_names_in_,
        "coef": model[-1].coef_,
        "abs_coef": abs(model[-1].coef_)
    }).sort_values("abs_coef")
    print(f"{model[-1].intercept_=}")
    print(results_df)
    print(f"{model[-1].alpha_=}")
    print(f"{train_mse=}")
    print(f"{test_mse=}")
    