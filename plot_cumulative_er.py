import json
import numpy as np
import matplotlib.pyplot as plt

# ===== GROUP 1 =====
json_paths_g1 = [
    "./ER_outputs/teacher_er_record5/qry_prob_eigen.json",
    "./ER_outputs/student_er_record5/qry_prob_eigen.json",
    "./ER_outputs/baseline_er_record5/qry_prob_eigen.json",
]
labels = ["Teacher", "Student", "Baseline"]
colors = ["navy", "red", "darkgreen"]

# ===== GROUP 2 =====
json_paths_g2 = [
    "./ER_outputs/teacher_er_record3/qry_prob_eigen.json",
    "./ER_outputs/student_er_record3/qry_prob_eigen.json",
    "./ER_outputs/baseline_er_record3/qry_prob_eigen.json",
]


eta = 0.85
save_name = "Eigenvalue_energy"

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# ---- Define sample indices (0-based) ----
g1_samples = [1, 2]  # sample 2, 3
g2_samples = [1, 3]  # sample 2, 4


# =====================================================
# Row 1 → Group 1
# =====================================================
for col, sample_idx in enumerate(g1_samples):

    ax = axes[0, col]
    ax.set_title(f"CLS dataset — Sample {col+1}")
    ax.set_xlabel("Eigenvalue Index (i)")
    ax.set_ylabel("Cumulative Energy Ratio")

    k_values = []

    for path, label, color in zip(json_paths_g1, labels, colors):

        with open(path, "r") as f:
            data = json.load(f)

        eigs = np.array(data[sample_idx][0])
        eigs = np.sort(eigs)[::-1]

        indices = np.arange(1, len(eigs) + 1)
        cum_energy = np.cumsum(eigs) / np.sum(eigs)

        k = np.argmax(cum_energy >= eta) + 1
        k_values.append(k)
        if label == "Teacher":
            teacher_k = k
        # Plot only up to k
        ax.plot(indices, cum_energy,
                color=color,
                linewidth=2,
                label=label)

    # if teacher_k is not None:
    #     ax.axvline(teacher_k,
    #                linestyle="--",
    #                color="royalblue",
    #                linewidth=2,
    #                alpha=0.8,
    #                label="0.85 energy")

    ax.set_ylim(0.4, 1.02)
    #ax.set_xlim(1, max(k_values))
    ax.grid(True)


# =====================================================
# Row 2 → Group 2
# =====================================================
for col, sample_idx in enumerate(g2_samples):

    ax = axes[1, col]
    ax.set_title(f"VQA dataset— Sample {col+1}")
    ax.set_xlabel("Eigenvalue Index (i)")
    ax.set_ylabel("Cumulative Energy Ratio")

    k_values = []

    for path, label, color in zip(json_paths_g2, labels, colors):

        with open(path, "r") as f:
            data = json.load(f)

        eigs = np.array(data[sample_idx][0])
        eigs = np.sort(eigs)[::-1]

        indices = np.arange(1, len(eigs) + 1)
        cum_energy = np.cumsum(eigs) / np.sum(eigs)

        k = np.argmax(cum_energy >= eta) + 1
        k_values.append(k)
        if label == "Teacher":
            teacher_k = k
        # Plot only up to k
        ax.plot(indices, cum_energy,
                color=color,
                linewidth=2,
                label=label)

    # if teacher_k is not None:
    #     ax.axvline(teacher_k,
    #                linestyle="--",
    #                color="royalblue",
    #                linewidth=2,
    #                alpha=0.8,
    #                label="0.85 energy")

    ax.set_ylim(0.4, 1.02)
    #ax.set_xlim(1, max(k_values))
    ax.grid(True)


# ---- Global Legend ----
handles, labels = axes[0, 0].get_legend_handles_labels()
unique = dict(zip(labels, handles))

fig.legend(unique.values(), unique.keys(),
           loc="upper center",
           ncol=4,
           bbox_to_anchor=(0.5, 0.95),
           frameon=False)

fig.subplots_adjust(hspace=0.35, top=0.88)
plt.savefig(f"{save_name}.png", dpi=300, bbox_inches="tight")
plt.show()