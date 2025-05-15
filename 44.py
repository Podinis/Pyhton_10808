import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9,  1.])
prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=3)

plt.plot(prob_pred, prob_true, marker='o')
plt.xlabel('Mean predicted value')  
plt.ylabel('Fraction of positives')
plt.title('Calibration plot')   
plt.grid()
plt.show()

print(prob_true)
print(prob_pred)