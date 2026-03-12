"""
Plot DDP original trajectory vs fitted waypoint trajectory side-by-side.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
from scenario import SCENARIO

COMP_DIR = os.path.dirname(__file__)
DDP_DIR = os.path.join(COMP_DIR, "results_ddp")
FIT_DIR = os.path.join(COMP_DIR, "results_ddp_fit")
CVAE_DIR = os.path.join(COMP_DIR, "results_cvae")
PLOT_DIR = os.path.join(COMP_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

n_q = SCENARIO["n_q"]


def load_csv(path):
    return np.genfromtxt(path, delimiter=",", names=True)


def main():
    ddp_q = load_csv(os.path.join(DDP_DIR, "q_traj.csv"))
    ddp_qd = load_csv(os.path.join(DDP_DIR, "qd_traj.csv"))
    fit_q = load_csv(os.path.join(FIT_DIR, "q_traj.csv"))
    cvae_q = load_csv(os.path.join(CVAE_DIR, "q_traj.csv"))
    cvae_qd = load_csv(os.path.join(CVAE_DIR, "qd_traj.csv"))

    # Waypoint locations (at segment boundaries: t=2.5, 5.0, 7.5)
    wp_times = [2.5, 5.0, 7.5]
    wp_vals = np.genfromtxt(os.path.join(FIT_DIR, "waypoints.csv"), delimiter=",", skip_header=1)
    wp_vals = wp_vals.reshape(SCENARIO["num_waypoints"], n_q)

    # ── Figure 1: Joint angles (DDP vs Fit vs CVAE, 3-way) ──
    fig, axes = plt.subplots(n_q, 1, figsize=(12, 2.2 * n_q), sharex=True)
    for i in range(n_q):
        jname = f"J{i+1}"
        axes[i].plot(ddp_q['t'], ddp_q[jname], 'b-', label='DDP', linewidth=1.5)
        axes[i].plot(fit_q['t'], fit_q[jname], 'g--', label='DDP->WP (fit)', linewidth=1.5)
        axes[i].plot(cvae_q['t'], cvae_q[jname], 'r:', label='CVAE', linewidth=1.5)
        # Waypoint markers
        for k, wt in enumerate(wp_times):
            axes[i].plot(wt, wp_vals[k, i], 'gD', markersize=6, zorder=5)
        axes[i].set_ylabel(f"J{i+1} (rad)")
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].legend(loc='upper right', fontsize=9)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Joint Angles: DDP vs Fitted Waypoints vs CVAE", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "fit_joint_angles.png"), dpi=150)
    print(f"Saved: {PLOT_DIR}/fit_joint_angles.png")

    # ── Figure 2: Fitting error per joint ──
    # Align lengths (DDP=101, fit=100)
    t_fit = fit_q['t']
    n_fit = len(t_fit)
    fig2, axes2 = plt.subplots(n_q, 1, figsize=(12, 2.2 * n_q), sharex=True)
    for i in range(n_q):
        jname = f"J{i+1}"
        ddp_interp = np.interp(t_fit, ddp_q['t'], ddp_q[jname])
        err = fit_q[jname] - ddp_interp
        axes2[i].plot(t_fit, np.rad2deg(err), 'g-', linewidth=1.2)
        axes2[i].axhline(0, color='k', linewidth=0.5)
        axes2[i].set_ylabel(f"J{i+1} err (deg)")
        axes2[i].grid(True, alpha=0.3)
    axes2[-1].set_xlabel("Time (s)")
    fig2.suptitle("Fitting Error: Quintic Spline - DDP (per joint)", fontsize=14)
    fig2.tight_layout()
    fig2.savefig(os.path.join(PLOT_DIR, "fit_error.png"), dpi=150)
    print(f"Saved: {PLOT_DIR}/fit_error.png")

    # ── Figure 3: Summary comparison (3-way) with orientation ──
    ddp_euler = load_csv(os.path.join(DDP_DIR, "euler_traj.csv"))
    cvae_euler = load_csv(os.path.join(CVAE_DIR, "euler_traj.csv"))

    target_ypr_deg = [SCENARIO["goal_euler_deg"][2],
                      SCENARIO["goal_euler_deg"][1],
                      SCENARIO["goal_euler_deg"][0]]

    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 9))

    # (0,0) All joint angles overlaid
    ax = axes3[0, 0]
    for i in range(n_q):
        jname = f"J{i+1}"
        l, = ax.plot(ddp_q['t'], ddp_q[jname], '-', alpha=0.7, linewidth=1)
        c = l.get_color()
        ax.plot(fit_q['t'], fit_q[jname], '--', color=c, alpha=0.7, linewidth=1)
        ax.plot(cvae_q['t'], cvae_q[jname], ':', color=c, alpha=0.5, linewidth=1)
    ax.set_title("Joint Angles (solid=DDP, dashed=Fit, dotted=CVAE)")
    ax.set_ylabel("rad")
    ax.grid(True, alpha=0.3)

    # (0,1) Orientation (DDP vs CVAE only, fit doesn't have euler)
    ax = axes3[0, 1]
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    euler_labels = ["Yaw", "Pitch", "Roll"]
    euler_keys = ["yaw", "pitch", "roll"]
    for i, (lbl, key) in enumerate(zip(euler_labels, euler_keys)):
        ax.plot(ddp_euler['t'], np.rad2deg(ddp_euler[key]), '-', color=colors[i], label=f'{lbl} DDP')
        ax.plot(cvae_euler['t'], np.rad2deg(cvae_euler[key]), '--', color=colors[i], label=f'{lbl} CVAE')
        ax.axhline(target_ypr_deg[i], color=colors[i], linestyle=':', alpha=0.4)
    ax.set_title("Body Orientation (DDP vs CVAE)")
    ax.set_ylabel("deg")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # (1,0) Fitting error heatmap-style (all joints)
    ax = axes3[1, 0]
    for i in range(n_q):
        jname = f"J{i+1}"
        ddp_interp = np.interp(t_fit, ddp_q['t'], ddp_q[jname])
        err_deg = np.rad2deg(fit_q[jname] - ddp_interp)
        ax.plot(t_fit, err_deg, linewidth=1.2, label=f"J{i+1}")
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_title("Fit Error per Joint (Spline - DDP)")
    ax.set_ylabel("Error (deg)")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

    # (1,1) Summary table
    ax = axes3[1, 1]
    ax.axis('off')

    # Load metas
    def load_meta(path):
        m = {}
        with open(path) as f:
            next(f)
            for line in f:
                k, v = line.strip().split(",", 1)
                try: m[k] = float(v)
                except: m[k] = v
        return m

    ddp_meta = load_meta(os.path.join(DDP_DIR, "meta.csv"))
    cvae_meta = load_meta(os.path.join(CVAE_DIR, "meta.csv"))
    fit_meta = load_meta(os.path.join(FIT_DIR, "meta.csv"))

    table_data = [
        ["Metric", "DDP", "DDP->WP", "CVAE"],
        ["Orient cost",
         f"{ddp_meta['orient_cost']:.4f}",
         f"{fit_meta['physics_loss']:.4f}",
         f"{cvae_meta['physics_loss']:.4f}"],
        ["Angle err",
         "~0",
         f"{fit_meta['angle_err_deg']:.2f}",
         f"{cvae_meta['angle_err_deg']:.2f}"],
        ["Fit MSE", "--", f"{fit_meta['fit_mse']:.6f}", "--"],
        ["Time (s)",
         f"{ddp_meta['elapsed_s']:.1f}",
         "--",
         f"{cvae_meta['elapsed_s']:.1f}"],
    ]
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    for j in range(4):
        table[0, j].set_text_props(fontweight='bold')
    ax.set_title("Summary", fontsize=12)

    fig3.suptitle("DDP vs Fitted Waypoints vs CVAE", fontsize=14)
    fig3.tight_layout()
    fig3.savefig(os.path.join(PLOT_DIR, "fit_summary.png"), dpi=150)
    print(f"Saved: {PLOT_DIR}/fit_summary.png")

    plt.close('all')
    print("Done.")


if __name__ == "__main__":
    main()
