import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Load the data
data = np.loadtxt("prices.txt")
df = pd.DataFrame(data)  # rows = time, columns = assets

# Initial parameters
initial_asset = 0
initial_start = 0
initial_end = len(df) - 1

# Create the figure and axis
fig, ax = plt.subplots(figsize=(12, 6))
plt.subplots_adjust(left=0.1, bottom=0.3)

# Initial plot
[line] = ax.plot(df.index, df[initial_asset], lw=2)
ax.set_title(f"Asset {initial_asset + 1} Prices")
ax.set_xlabel("Time (Days)")
ax.set_ylabel("Price")
ax.grid(True)

# Create sliders
ax_asset = plt.axes([0.1, 0.2, 0.65, 0.03])
ax_start = plt.axes([0.1, 0.15, 0.65, 0.03])
ax_end = plt.axes([0.1, 0.1, 0.65, 0.03])

slider_asset = Slider(ax_asset, "Asset", 0, df.shape[1] - 1, valinit=initial_asset, valstep=1)
slider_start = Slider(ax_start, "Start", 0, len(df) - 2, valinit=initial_start, valstep=1)
slider_end = Slider(ax_end, "End", 1, len(df) - 1, valinit=initial_end, valstep=1)

# Update function
def update(val):
    asset = int(slider_asset.val)
    start = int(slider_start.val)
    end = int(slider_end.val)
    if end <= start:
        return
    line.set_ydata(df.iloc[start:end+1, asset])
    line.set_xdata(range(start, end+1))
    ax.set_xlim(start, end)
    ax.set_ylim(df.iloc[start:end+1, asset].min() * 0.95,
                df.iloc[start:end+1, asset].max() * 1.05)
    ax.set_title(f"Asset {asset + 1} Prices (Day {start} to {end})")
    fig.canvas.draw_idle()

# Connect sliders to update
slider_asset.on_changed(update)
slider_start.on_changed(update)
slider_end.on_changed(update)

plt.show()
