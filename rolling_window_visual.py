import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_rolling_windows(train_size=252, val_size=42, step=21, n_folds=3):
    fig, ax = plt.subplots(figsize=(10, 2.5))

    # Timeline
    total_length = train_size + val_size + (n_folds - 1) * step
    ax.set_xlim(0, total_length)
    ax.set_ylim(0, 3)
    ax.set_yticks([])
    ax.set_xticks([])

    # Colors
    train_color = "#1f77b4"
    val_color = "#ff7f0e"

    for i in range(n_folds):
        start_train = i * step
        end_train = start_train + train_size
        start_val = end_train
        end_val = start_val + val_size

        # Draw train window
        ax.add_patch(patches.Rectangle((start_train, 1), train_size, 0.5, color=train_color))
        # Draw val window
        ax.add_patch(patches.Rectangle((start_val, 1), val_size, 0.5, color=val_color))

        # Labels
        ax.text(start_train + train_size / 2, 1.6, f"Train {i+1}", ha='center', va='bottom', fontsize=9)
        ax.text(start_val + val_size / 2, 1.6, f"Val {i+1}", ha='center', va='bottom', fontsize=9)

    # Legend
    train_patch = patches.Patch(color=train_color, label='Training Window')
    val_patch = patches.Patch(color=val_color, label='Validation Window')
    ax.legend(handles=[train_patch, val_patch], loc='upper right')

    plt.title("Rolling Window Cross-Validation Timeline")
    plt.tight_layout()
    plt.show()
    plt.savefig("data/tuning_results/figures", dpi=300, bbox_inches='tight')

plot_rolling_windows(train_size=252, val_size=42, step=21, n_folds=3)