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
save_name = "text_spectrum_comparison"
# ===================

plt.figure(figsize=(9,6))
ax1 = plt.gca()
ax1.set_yscale("log")
ax1.set_xlabel("Eigenvalue Index (i)")
ax1.set_ylabel("Eigenvalue Magnitude")

for json_path, label, color in zip(json_paths, labels, colors):

    with open(json_path, "r") as f:
        data = json.load(f)

    eigs = np.array(data[2][0])   # vision eigenvalues
    eigs = np.sort(eigs)[::-1]

    indices = np.arange(1, len(eigs) + 1)
    cum_energy = np.cumsum(eigs) / np.sum(eigs)
    k = np.argmax(cum_energy >= eta) + 1

    # Plot spectrum
    ax1.plot(indices, eigs, color=color, label=f"{label} (k={k})")

    # Optional: mark threshold position
    #ax1.axvline(k, linestyle="--", color=color, alpha=0.4)

ax1.legend()
ax1.grid(True)

plt.title(f"Eigenvalue Spectrum Comparison")
plt.tight_layout()
plt.savefig(f"{save_name}.png", dpi=300, bbox_inches="tight")
plt.show()