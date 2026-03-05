import json
import numpy as np
import matplotlib.pyplot as plt

# ===== CONFIG =====
json_paths = [
    "./ER_outputs/teacher_er_record2/qry_prob_eigen.json",
    "./ER_outputs/student_er_record2/qry_prob_eigen.json",
    "./ER_outputs/baseline_er_record2/qry_prob_eigen.json",
]

labels = ["Teacher", "Student", "Baseline"]
colors = ["royalblue", "darkorange", "forestgreen"]

eta = 0.85
save_name = "eigenvalue_energy_comparison"
# ===================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i in range(4):   # for data[2][0..3]

    ax = axes[i]

    ax.set_title(f"Eigenvalue energy comparison for sample {i}")
    ax.set_xlabel("Eigenvalue Index (i)")
    ax.set_ylabel("Cumulative Energy Ratio")
    ax.set_ylim(0, 1.02)

    for json_path, label, color in zip(json_paths, labels, colors):

        with open(json_path, "r") as f:
            data = json.load(f)

        eigs = np.array(data[2][i])
        eigs = np.sort(eigs)[::-1]

        indices = np.arange(1, len(eigs) + 1)
        cum_energy = np.cumsum(eigs) / np.sum(eigs)
        k = np.argmax(cum_energy >= eta) + 1

        ax.plot(indices, cum_energy,
                color=color,
                linewidth=2,
                alpha=0.8,
                label=f"{label} (k={k})")

    ax.grid(True)

# One global legend
handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels,
           loc="upper center",
           ncol=3,
           bbox_to_anchor=(0.5, 0.98))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"{save_name}.png", dpi=300, bbox_inches="tight")
#plt.savefig(f"{save_name}.pdf", bbox_inches="tight")
plt.show()