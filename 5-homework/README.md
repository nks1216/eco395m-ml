# Homework 5 (10 Points)

In your private repo called `eco395m-ml-student` and invite the TA and I. Make a folder called `5-homework`. For exercises containing programming, take the starter code (if there is any) and move it into the folder and solve. For written exercises (including mathematical ones) either submit the solution as a neatly written scan or picture or a markdown file (containing LaTeX). Inside a file called `README.md` in the homework folder, indicate which file, images etc. contain which solutions.

1. **Micro-Weighted F1 is Accuracy for Binary Classification**: (1 Point) Show that for a binary classification problem, micro-weighted F1 is equivalent to accuracy.  

2. **Balanced Accuracy is Accuracy on Balanced Dataset**: (1 Point) Show that Balanced Accuracy=(1/2)(recall + specificity) reduces to Accuracy if the classes are balanced for a binary classification problem.  

3. **Balanced Accuracy is Accuracy After Balancing**: (2 Points) Show that reweighting the 0-class observations (TN, FP) by the imbalance and computing accuracy is equivalent to Balanced Accuracy=(1/2)(recall + specificity) for a binary classification problem.


4. **ROC and Precision/Recall Curves**: (4 Points) Implement ROC and Precision-Recall Curves using only Python and, optionally, numpy. Show your results with matplotlib. Demonstrate that it works correctly with an example (you may use the example we did in class).

5. **AUC and Average Precision**: (2 Points) Using only Python and, optionally, numpy, implement ROC-AUC and average precision using your solution to the last problem. Demonstrate that it works correctly with an example (you may use the example we did in class).