from matplotlib import pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

y_score = [1, 2, 3, 4, 5, 6]
y_true = [0, 0, 1, 0, 1, 1]

RocCurveDisplay.from_predictions(y_true=y_true, y_score=y_score)
plt.savefig("roc.png")

PrecisionRecallDisplay.from_predictions(y_true=y_true, y_score=y_score)
plt.savefig("pr.png")


