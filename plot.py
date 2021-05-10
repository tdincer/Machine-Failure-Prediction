import os
import sys
import utility as ut
from joblib import load
import matplotlib.pyplot as plt
from matplotlib import animation as an


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
plt.tight_layout()
ax5.set_xlabel('Sample No')

ax0.set_ylim(-0.25, 1.25)
ax0.set_xlim(0, NTIME)
ax0.set_ylabel('Failure')
line0, = ax0.plot([], [], lw=1)

ax1.set_ylim(5, 78)
ax1.set_xlim(0, NTIME)
ax1.set_ylabel('Torque')
line1, = ax1.plot([], [], lw=1)

ax2.set_ylim(305, 315)
ax2.set_xlim(0, NTIME)
ax2.set_ylabel(r'T$_{Process}$')
line2, = ax2.plot([], [], lw=1)

ax3.set_ylim(290, 310)
ax3.set_xlim(0, NTIME)
ax3.set_ylabel(r'T$_{Air}$')
line3, = ax3.plot([], [], lw=1)

ax4.set_ylim(1180, 2675)
ax4.set_xlim(0, NTIME)
ax4.set_ylabel(r'V$_{Rot}$')
line4, = ax4.plot([], [], lw=1)

ax5.set_ylim(-0.25, 250)
ax5.set_xlim(0, NTIME)
ax5.set_ylabel('Tool \n  Wear')
line5, = ax5.plot([], [], lw=1)

ax6.set_ylim(-0.25, 1.25)
ax6.set_xlim(0, NTIME)
ax6.set_ylabel('Probability')
line6, = ax6.plot([], [], lw=1)


def animate(i):
    x = y_test.loc[0:i].index

    y = y_test.loc[0:i]
    line0.set_data(x, y)

    y = x_test.loc[0:i, 'Torque [Nm]']
    line1.set_data(x, y)

    y = x_test.loc[0:i, 'Process temperature [K]']
    line2.set_data(x, y)

    y = x_test.loc[0:i, 'Air temperature [K]']
    line3.set_data(x, y)

    y = x_test.loc[0:i, 'Rotational speed [rpm]']
    line4.set_data(x, y)

    y = x_test.loc[0:i, 'Tool wear [min]']
    line5.set_data(x, y)

    y = model.predict_proba(x_test.loc[0:i])
    line6.set_data(x, y)
    return line0, line1, line2, line3, line4, line5, line6


anim = an.FuncAnimation(fig, animate, frames=len(y_test), interval=1, blit=True)

Writer = an.writers['ffmpeg']
writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)

file_name = os.path.join('VISUALS', 'anim.mp4')
anim.save(file_name, writer=writer, dpi=500)
