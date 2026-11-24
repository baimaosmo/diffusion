"""Betti-curve summaries on the complete D=1-FC filtration over [0,2].

The caller supplies finite H0/H1 diagrams. The single essential H0 class
removed for diagram matching is restored when counting connected components.
"""

import numpy as np


def event_grid(groups, bounds=(0., 2.)):
    values = [np.asarray(bounds)]
    for diagrams in groups:
        for sample in diagrams:
            for diagram in sample:
                points = np.asarray(diagram).ravel()
                values.append(points[np.isfinite(points)])
    grid = np.unique(np.concatenate(values))
    return grid[(grid >= bounds[0]) & (grid <= bounds[1])]


def mean_betti_curve(diagrams, grid, dimension):
    """Right-continuous mean count at each epsilon: birth <= epsilon < death."""
    if not diagrams or dimension not in (0, 1):
        raise ValueError("Need at least one diagram and homology dimension 0 or 1.")
    grid = np.asarray(grid, dtype=float)
    result = np.zeros(len(grid), dtype=float)
    for sample in diagrams:
        points = np.asarray(sample[dimension], dtype=float).reshape(-1, 2)
        if not np.isfinite(points).all():
            raise ValueError("Supply only finite diagrams; essential H0 is restored separately.")
        result += (np.searchsorted(np.sort(points[:, 0]), grid, side="right")
                   - np.searchsorted(np.sort(points[:, 1]), grid, side="right"))
    result /= len(diagrams)
    if dimension == 0:
        result += (grid >= 0).astype(float)
    return result


def integrated_betti_error(real, generated, dimension):
    """Exact mean absolute gap between group mean step curves over [0,2].

    Units: component/cycle count. This is not subject-wise pairing or a
    full distribution distance. All birth/death events are used, without
    grid approximation or trapezoidal interpolation through discontinuities.
    """
    grid = event_grid([real, generated])
    gap = np.abs(mean_betti_curve(real, grid, dimension)
                 - mean_betti_curve(generated, grid, dimension))
    return float(np.sum(gap[:-1] * np.diff(grid)) / (grid[-1] - grid[0]))


def save_persistence_figure(unit_diagrams, methods, ph_distances, output, class_id):
    """Render primary PH-fidelity figure only when invoked by the plot entrypoint."""
    import csv
    import matplotlib.pyplot as plt

    groups = ["Real", *methods]
    colors = ["#555555", "#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    grid = event_grid([unit["diagrams"][g] for unit in unit_diagrams for g in groups])
    # dimensions: unit, group, homology dimension, filtration event
    curves = np.array([[[mean_betti_curve(unit["diagrams"][g], grid, dim)
                         for dim in [0, 1]] for g in groups] for unit in unit_diagrams])
    errors = np.empty((len(unit_diagrams), len(methods), 2))
    records = []
    for u, unit in enumerate(unit_diagrams):
        for m, name in enumerate(methods):
            for dim in [0, 1]:
                errors[u, m, dim] = integrated_betti_error(unit["diagrams"]["Real"], unit["diagrams"][name], dim)
                records.append({"unit": unit["id"], "method": name, "homology": f"H{dim}",
                                "samples_per_group": len(unit["diagrams"][name]),
                                "integrated_mean_curve_error": errors[u, m, dim]})
    with (output / "betti_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(output / "betti_curves.npz", epsilon=grid, curves=curves,
                        groups=np.array(groups), unit_ids=np.array([u["id"] for u in unit_diagrams]),
                        dimensions=np.array([0, 1]))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout="constrained")
    for dim in [0, 1]:
        ax = axes[0, dim]
        for g, name in enumerate(groups):
            values = curves[:, g, dim]
            mean = values.mean(axis=0)
            ax.step(grid, mean, where="post", color=colors[g], label=name, lw=1.8)
            if len(values) > 1:
                sd = values.std(axis=0, ddof=1)
                ax.fill_between(grid, np.maximum(0, mean-sd), mean+sd, step="post", color=colors[g], alpha=.12)
        ax.set(title=f"({'a' if dim == 0 else 'b'}) H{dim} Betti curve",
               xlabel="Filtration epsilon (D = 1 - FC)",
               ylabel="Mean connected-component count" if dim == 0 else "Mean H1 feature count", xlim=(0, 2))
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=9)
    for ax, values, title, ylabel in [
        (axes[1, 0], errors, "(c) Integrated Betti-curve error", "Average count difference (lower is closer)"),
        (axes[1, 1], ph_distances, "(d) Persistence-diagram set distance", "Mean matched diagram W1 (lower is closer)"),
    ]:
        width = .7 / len(methods)
        for m, name in enumerate(methods):
            pos = np.arange(2) + (m - (len(methods)-1)/2)*width
            data = values[:, m]
            ax.bar(pos, data.mean(axis=0), width=width*.85, color=colors[m+1], alpha=.8, label=name)
            if len(data) > 1:
                ax.errorbar(pos, data.mean(axis=0), yerr=data.std(axis=0, ddof=1),
                            fmt="none", color="#333333", capsize=3)
            for row in data:
                ax.scatter(pos, row, s=18, facecolors="white", edgecolors=colors[m+1], zorder=3)
        ax.set_xticks([0, 1], ["H0", "H1"])
        ax.set(title=title, ylabel=ylabel)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.2)
        ax.legend(frameon=False, fontsize=9)
    selected_counts = {g: sorted({len(u["diagrams"][g]) for u in unit_diagrams}) for g in groups}
    best_case = all(max(selected_counts[m]) == 1 for m in methods)
    prefix = "Best-generated persistent topology" if best_case else "Persistent topology fidelity"
    fig.suptitle(f"{prefix} | class {class_id}", fontsize=16, weight="bold")
    fig.supxlabel("Full weighted filtration; same sampled FCs as the PH-distance evaluation. Essential H0 included in Betti counts.\n"
                  + ("One generated FC per fold selected by training-centroid similarity; best-case descriptive analysis. " if best_case else "")
                  + "Mean +/- SD across evaluation units (descriptive, not CI). Fidelity alone does not establish the effect of PH guidance.", fontsize=9)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(output / f"persistence_fidelity.{ext}", dpi=250)
    plt.close(fig)
