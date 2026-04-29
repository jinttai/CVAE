"""Plot CVAE vs MLP training loss curves in one graph."""
import pandas as pd
import matplotlib.pyplot as plt

# Load data
cvae_df = pd.read_csv(r"C:\CVAE\outputs\plots\cvae_training_curve\v4.csv")
mlp_df = pd.read_csv(r"C:\CVAE\outputs\plots\mlp_training_curve\v4.csv")

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(cvae_df["epoch"], cvae_df["train_loss"], label="CVAE", alpha=0.85)
ax.plot(mlp_df["epoch"], mlp_df["train_loss"], label="MLP", alpha=0.85)

ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("Train Loss", fontsize=13)
ax.set_title("Training Loss: CVAE vs MLP", fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()

out_path = r"C:\CVAE\outputs\plots\cvae_vs_mlp_train_loss.png"
fig.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")
plt.show()
