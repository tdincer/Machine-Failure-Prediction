import os
import sys
import utility as ut
from joblib import load
import matplotlib.pyplot as plt


# Data
_, _, x_test, y_test, _, _ = ut.get_train_test_data()
y_test = y_test[x_test['Type'] == 'L']
x_test = x_test[x_test['Type'] == 'L']
x_test = x_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Model
sys.path.insert(1, 'MODELS')
model = load('./MODELS/lgb.model')

NTIME = len(y_test)

fig, (ax6, ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(7, sharex=True, figsize=(8, 7),
                                                        gridspec_kw={'hspace': 0.2, 'wspace': 0.1})


x = y_test.index
ax0.plot(x, y_test)
ax1.plot(x, x_test['Torque [Nm]'])
ax2.plot(x, x_test['Process temperature [K]'])
ax3.plot(x, x_test['Air temperature [K]'])
ax4.plot(x, x_test['Rotational speed [rpm]'])
ax5.plot(x, x_test['Tool wear [min]'])
ax6.plot(x, model.predict_proba(x_test))

plt.tight_layout()
ax5.set_xlabel('Sample No')

ax0.set_ylim(-0.25, 1.25)
ax0.set_xlim(0, NTIME)
ax0.set_ylabel('Failure')

ax1.set_ylim(5, 78)
ax1.set_xlim(0, NTIME)
ax1.set_ylabel('Torque')

ax2.set_ylim(305, 315)
ax2.set_xlim(0, NTIME)
ax2.set_ylabel(r'T$_{Process}$')

ax3.set_ylim(290, 310)
ax3.set_xlim(0, NTIME)
ax3.set_ylabel(r'T$_{Air}$')

ax4.set_ylim(1180, 2675)
ax4.set_xlim(0, NTIME)
ax4.set_ylabel(r'V$_{Rot}$')

ax5.set_ylim(-0.25, 250)
ax5.set_xlim(0, NTIME)
ax5.set_ylabel('Tool \n  Wear')

ax6.set_ylim(-0.25, 1.25)
ax6.set_xlim(0, NTIME)
ax6.set_ylabel('Prediction')

file_name = os.path.join('VISUALS', 'result.png')
plt.savefig(file_name)
