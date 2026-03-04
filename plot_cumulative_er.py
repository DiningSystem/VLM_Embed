import json
import numpy as np
import matplotlib.pyplot as plt

# ===== CONFIG =====
json_paths = [
    "./ER_outputs/teacher_er_record/qry_prob_eigen.json",
    "./ER_outputs/trained_student_er_record/qry_prob_eigen.json",
    "./ER_outputs/student_constrastive_er_record/qry_prob_eigen.json",
]

labels = ["Teacher", "Student", "Baseline"]
colors = ["royalblue", "darkorange", "forestgreen"]

eta = 0.85
save_name = "cumulative_comparison"
plt.figure(figsize=(9,6))
ax = plt.gca()

ax.set_xlabel("Eigenvalue Index (i)")
ax.set_ylabel("Cumulative Energy Ratio")
ax.set_ylim(0, 1.02)

for json_path, label, color in zip(json_paths, labels, colors):

    with open(json_path, "r") as f:
        data = json.load(f)

    eigs = np.array(data[2][0])
    eigs = np.sort(eigs)[::-1]

    indices = np.arange(1, len(eigs) + 1)
    cum_energy = np.cumsum(eigs) / np.sum(eigs)
    k = np.argmax(cum_energy >= eta) + 1

    ax.plot(indices, cum_energy,
            color=color,
            linewidth=2,
            alpha=0.7,
            label=f"{label} (k={k})")

    #ax.axhline(eta, linestyle="--", color="black", alpha=0.5)
    #ax.axvline(k, linestyle=":", color=color, alpha=0.4)

ax.legend()
ax.grid(True)
plt.title(f"Cumulative Energy Comparison")
plt.tight_layout()
plt.savefig("cumulative_only.png", dpi=300, bbox_inches="tight")

plt.show()