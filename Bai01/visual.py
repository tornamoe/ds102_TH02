import matplotlib.pyplot as plt


def plot_loss(loss_history, title="Loss theo số vòng lặp", save_path=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, len(loss_history) + 1), loss_history, linewidth=2, color="tab:red")
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cross-entropy loss")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()
