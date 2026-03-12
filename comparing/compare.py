"""
Compare DDP and CVAE results side-by-side.
Run run_ddp.py and run_cvae.py first, then run this script.
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
CVAE_DIR = os.path.join(COMP_DIR, "results_cvae")


def load_csv(path):
    return np.genfromtxt(path, delimiter=",", names=True)


def load_meta(path):
    meta = {}
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            k, v = line.strip().split(",", 1)
            try:
                meta[k] = float(v)
            except ValueError:
                meta[k] = v
    return meta


def main():
    # Check results exist
    for d, name in [(DDP_DIR, "DDP"), (CVAE_DIR, "CVAE")]:
        if not os.path.isdir(d):
            print(f"ERROR: {name} results not found at {d}")
            print(f"  Run 'python comparing/run_ddp.py' and 'python comparing/run_cvae.py' first.")
            return

    # Load data
    ddp_q = load_csv(os.path.join(DDP_DIR, "q_traj.csv"))
    ddp_qd = load_csv(os.path.join(DDP_DIR, "qd_traj.csv"))
    ddp_euler = load_csv(os.path.join(DDP_DIR, "euler_traj.csv"))
    ddp_meta = load_meta(os.path.join(DDP_DIR, "meta.csv"))

    cvae_q = load_csv(os.path.join(CVAE_DIR, "q_traj.csv"))
    cvae_qd = load_csv(os.path.join(CVAE_DIR, "qd_traj.csv"))
    cvae_euler = load_csv(os.path.join(CVAE_DIR, "euler_traj.csv"))
    cvae_meta = load_meta(os.path.join(CVAE_DIR, "meta.csv"))

    n_q = SCENARIO["n_q"]
    goal_euler_rad = np.deg2rad(SCENARIO["goal_euler_deg"])  # [roll, pitch, yaw]
    # target in [yaw, pitch, roll] order for plotting
    target_ypr_deg = [SCENARIO["goal_euler_deg"][2],   # yaw
                      SCENARIO["goal_euler_deg"][1],   # pitch
                      SCENARIO["goal_euler_deg"][0]]   # roll

    save_dir = os.path.join(COMP_DIR, "plots")
    os.makedirs(save_dir, exist_ok=True)

    # ── Print summary ──
    print("=" * 60)
    print(f"{'':>25} {'DDP':>15} {'CVAE':>15}")
    print("=" * 60)
    print(f"{'Orient cost':>25} {ddp_meta.get('orient_cost', 'N/A'):>15.6f} {cvae_meta.get('physics_loss', 'N/A'):>15.6f}")
    if 'angle_err_deg' in cvae_meta:
        print(f"{'Angle error (deg)':>25} {'--':>15} {cvae_meta['angle_err_deg']:>15.4f}")
    print(f"{'Time (s)':>25} {ddp_meta['elapsed_s']:>15.3f} {cvae_meta['elapsed_s']:>15.3f}")
    print("=" * 60)

    # ── Figure 1: Joint Angles ──
    fig, axes = plt.subplots(n_q, 1, figsize=(12, 2.2 * n_q), sharex=True)
    for i in range(n_q):
        jname = f"J{i+1}"
        axes[i].plot(ddp_q['t'], ddp_q[jname], 'b-', label='DDP', linewidth=1.2)
        axes[i].plot(cvae_q['t'], cvae_q[jname], 'r--', label='CVAE', linewidth=1.2)
        axes[i].set_ylabel(f"J{i+1} (rad)")
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].legend(loc='upper right')
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Joint Angles: DDP vs CVAE", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "joint_angles.png"), dpi=150)
    print(f"Saved: {save_dir}/joint_angles.png")

    # ── Figure 2: Joint Velocities ──
    fig2, axes2 = plt.subplots(n_q, 1, figsize=(12, 2.2 * n_q), sharex=True)
    for i in range(n_q):
        jname = f"dJ{i+1}"
        axes2[i].plot(ddp_qd['t'], ddp_qd[jname], 'b-', label='DDP', linewidth=1.2)
        axes2[i].plot(cvae_qd['t'], cvae_qd[jname], 'r--', label='CVAE', linewidth=1.2)
        axes2[i].set_ylabel(f"dJ{i+1} (rad/s)")
        axes2[i].grid(True, alpha=0.3)
        if i == 0:
            axes2[i].legend(loc='upper right')
    axes2[-1].set_xlabel("Time (s)")
    fig2.suptitle("Joint Velocities: DDP vs CVAE", fontsize=14)
    fig2.tight_layout()
    fig2.savefig(os.path.join(save_dir, "joint_velocities.png"), dpi=150)
    print(f"Saved: {save_dir}/joint_velocities.png")

    # ── Figure 3: Body Orientation (Euler) ──
    fig3, axes3 = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    euler_labels = ["Yaw (Z)", "Pitch (Y)", "Roll (X)"]
    euler_keys = ["yaw", "pitch", "roll"]

    for i, (label, key) in enumerate(zip(euler_labels, euler_keys)):
        ddp_deg = np.rad2deg(ddp_euler[key])
        cvae_deg = np.rad2deg(cvae_euler[key])
        axes3[i].plot(ddp_euler['t'], ddp_deg, 'b-', label='DDP', linewidth=1.2)
        axes3[i].plot(cvae_euler['t'], cvae_deg, 'r--', label='CVAE', linewidth=1.2)
        axes3[i].axhline(target_ypr_deg[i], color='g', linestyle=':', linewidth=1.5, label=f'Target ({target_ypr_deg[i]:.1f})')
        axes3[i].set_ylabel(f"{label} (deg)")
        axes3[i].grid(True, alpha=0.3)
        axes3[i].legend(loc='upper right', fontsize=9)

    axes3[-1].set_xlabel("Time (s)")
    fig3.suptitle("Body Orientation: DDP vs CVAE", fontsize=14)
    fig3.tight_layout()
    fig3.savefig(os.path.join(save_dir, "orientation.png"), dpi=150)
    print(f"Saved: {save_dir}/orientation.png")

    # ── Figure 4: Combined summary ──
    fig4, axes4 = plt.subplots(2, 2, figsize=(14, 9))

    # (0,0) Joint angles overlaid (all joints)
    ax = axes4[0, 0]
    for i in range(n_q):
        jname = f"J{i+1}"
        l1, = ax.plot(ddp_q['t'], ddp_q[jname], '-', alpha=0.7, linewidth=1)
        ax.plot(cvae_q['t'], cvae_q[jname], '--', alpha=0.7, linewidth=1, color=l1.get_color())
    ax.set_title("Joint Angles (solid=DDP, dashed=CVAE)")
    ax.set_ylabel("rad")
    ax.grid(True, alpha=0.3)

    # (0,1) Orientation
    ax = axes4[0, 1]
    for i, (label, key) in enumerate(zip(euler_labels, euler_keys)):
        colors = ['tab:blue', 'tab:orange', 'tab:green']
        ax.plot(ddp_euler['t'], np.rad2deg(ddp_euler[key]), '-', color=colors[i], alpha=0.8, label=f'{label} DDP')
        ax.plot(cvae_euler['t'], np.rad2deg(cvae_euler[key]), '--', color=colors[i], alpha=0.8, label=f'{label} CVAE')
        ax.axhline(target_ypr_deg[i], color=colors[i], linestyle=':', alpha=0.4)
    ax.set_title("Body Orientation")
    ax.set_ylabel("deg")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # (1,0) Joint velocities
    ax = axes4[1, 0]
    for i in range(n_q):
        jname = f"dJ{i+1}"
        l1, = ax.plot(ddp_qd['t'], ddp_qd[jname], '-', alpha=0.7, linewidth=1)
        ax.plot(cvae_qd['t'], cvae_qd[jname], '--', alpha=0.7, linewidth=1, color=l1.get_color())
    ax.set_title("Joint Velocities (solid=DDP, dashed=CVAE)")
    ax.set_ylabel("rad/s")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)

    # (1,1) Summary table
    ax = axes4[1, 1]
    ax.axis('off')
    table_data = [
        ["Metric", "DDP", "CVAE"],
        ["Orient cost", f"{ddp_meta.get('orient_cost', 'N/A'):.6f}", f"{cvae_meta.get('physics_loss', 'N/A'):.6f}"],
        ["Time (s)", f"{ddp_meta['elapsed_s']:.2f}", f"{cvae_meta['elapsed_s']:.2f}"],
    ]
    if 'angle_err_deg' in cvae_meta:
        table_data.append(["Angle err (deg)", "--", f"{cvae_meta['angle_err_deg']:.4f}"])

    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    # Bold header row
    for j in range(3):
        table[0, j].set_text_props(fontweight='bold')
    ax.set_title("Summary", fontsize=12)

    fig4.suptitle(f"DDP vs CVAE Comparison (Goal: roll={SCENARIO['goal_euler_deg'][0]}, pitch={SCENARIO['goal_euler_deg'][1]}, yaw={SCENARIO['goal_euler_deg'][2]})", fontsize=13)
    fig4.tight_layout()
    fig4.savefig(os.path.join(save_dir, "summary.png"), dpi=150)
    print(f"Saved: {save_dir}/summary.png")

    plt.close('all')
    print("\nDone. All comparison plots saved.")


if __name__ == "__main__":
    main()
