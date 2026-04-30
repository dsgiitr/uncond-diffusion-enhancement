import json
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the provided JSON data
with open('efficientnet-b0_history.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# 2. Compute Cumulative Time
# Since "time" represents the time per epoch, calculating the cumulative sum 
# gives us the elapsed time in seconds as the training progressed.
df['cumulative_time'] = df['time'].cumsum()

# 3. Apply Exponentially Weighted Moving Average (EWMA)
# The 'span' parameter dictates how smooth the line will be. 
# A larger span considers more past epochs, making the line smoother.
span_value = 10 
df['val_smiou_smoothed'] = df['val_smiou'].ewm(span=span_value, adjust=False).mean()

# 4. Plotting Configuration
plt.figure(figsize=(12, 6))

# Plot the raw, noisy validation smiou underneath with some transparency (alpha)
# plt.plot(df['cumulative_time'], df['val_smiou'], 
#          label='Raw val_smiou', color='skyblue', alpha=0.5, linewidth=1.5)

# Plot the smoothed EWMA line on top
plt.plot(df['epoch'], df['val_smiou_smoothed'], 
         label=f'Validation mIoU', color='darkblue', linewidth=2.5)

# Add labels, title, and styling
plt.title('Validation MIOU Trajectory over Training Epochs', fontsize=18, pad=15)
plt.xlabel('Number of Epochs', fontsize=16)
plt.ylabel('Validation MIOU', fontsize=16)

# Grid and legend
plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend(loc='best', fontsize=11)
plt.tight_layout()

# 5. Save and Display the Plot
plt.savefig('validation_miou_trajectory.pdf', format='pdf', bbox_inches='tight')
plt.show()