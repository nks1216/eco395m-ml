# Homework 6 (10 Points + 3 Bonus)

In your private repo called eco395m-ml-student and invite the TA and I. Make a folder called 6-homework. For exercises containing programming, take the starter code (if there is any) and move it into the folder and solve. For written exercises (including mathematical ones) either submit the solution as a neatly written scan or picture or a markdown file (containing LaTeX). Inside a file called README.md in the homework folder, indicate which file, images etc. contain which solutions.

1. **Scikit-learn Logistic Regression**: (2 Points) Scikit-learn's Logistic Regression, with the default parameters, doesn’t match other packages/implementations. In particular, by default, the objective function is slightly different from standard logistic regression. How? What option can you change to get the standard behavior?

2. **Logistic Regression**: (6 Points) Implement Logistic Regression with gradient descent using only Python and Numpy, and write tests to demonstrate that it is working as intended.

3. **Scikit-learn Random Forest**: (2 Points) In Scikit-Learn, does RandomForestClassifier use voting or average leaf probabilities?

Stacking v.s. Ensemble in Random Forest

(3 Points): Its possible to use a stacking estimator as the aggregation step in a Random Forest rather than voting or averaging class probabilities over trees. Implement this model using scikit-learn by passing a list of copies of a DecisionTreeClassifier or ExtraTreeClassifier to StackingClassifier and use Logistic Regression as the final estimator. Investigate the performance across datasets. A hypothesis is that performance will likely be similar to the Random Forest. However, probabilities from the model may be better calibrated because Logistic Regression tends to produce well calibrated probabilities and we use it as the final estimator (See https://scikit-learn.org/stable/modules/calibration.html). Investigate and discuss.
