import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


if __name__ == "__main__":

    CAT_VARS = ["League", "Division", "NewLeague"]

    df = pd.read_csv("data/hitters/hitters.csv").dropna(how="any")


    NUM_VARS = [col for col in df.columns if not col in CAT_VARS and col != "Salary"]

    y = df["Salary"].values
    X = df.drop("Salary", axis="columns")
    
    print(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    ct = ColumnTransformer(
        [
            ("cat_preprocess", OrdinalEncoder(), CAT_VARS),
            ("num_preprocess", "passthrough", NUM_VARS)
        ]

    )
    model = make_pipeline(ct, RandomForestRegressor(n_estimators=1000)).set_output(transform="pandas")

    model.fit(X_train, y_train)
    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_hat_train)
    test_mse = mean_squared_error(y_test, y_hat_test)


    # results_df = pd.DataFrame({
    #     "feature": model[-1].feature_names_in_,
    #     "coef": model[-1].coef_,
    #     "abs_coef": abs(model[-1].coef_)
    # }).sort_values("abs_coef")
    # print(f"{model[-1].intercept_=}")
    # print(results_df)
    print(f"{train_mse=}")
    print(f"{test_mse=}")
    