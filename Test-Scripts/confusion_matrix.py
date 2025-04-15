import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

# Load the CSV
df = pd.read_csv("classification_results.csv")

# Get unique class names (in order)
class_names = sorted(df["true_class"].unique())

# Build raw confusion matrix
y_true = df["true_class"]
y_pred = df["predicted_class"]
cm = confusion_matrix(y_true, y_pred, labels=class_names)

# Build normalized confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

def plot_cm(cm, labels, title, filename, normalized=False):
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    ax = sns.heatmap(
        cm, annot=True, fmt=".2f" if normalized else "d",
        xticklabels=labels, yticklabels=labels,
        cmap="Blues", cbar=True, square=True, linewidths=0.5, linecolor='gray'
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {title.lower()} to {filename}")

# Plot raw
plot_cm(cm, class_names, "Confusion Matrix (Raw Counts)", "confusion_matrix_pretty_raw.png", normalized=False)

# Plot normalized
plot_cm(cm_normalized, class_names, "Confusion Matrix (Normalized)", "confusion_matrix_pretty_normalized.png", normalized=True)
