import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

# Add root directory to sys.path to find src
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

from src.models.cvae import CVAE  # noqa: E402
from src.training.physics_layer import PhysicsLayer  # noqa: E402
from src.dynamics.urdf2robot_torch import urdf2robot  # noqa: E402


def euler_to_quaternion(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion (x, y, z, w)
    Using ZYX convention (yaw around Z, pitch around Y, roll around X)
    """
    cr = torch.cos(roll / 2)
    sr = torch.sin(roll / 2)
    cp = torch.cos(pitch / 2)
    sp = torch.sin(pitch / 2)
    cy = torch.cos(yaw / 2)
    sy = torch.sin(yaw / 2)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy

    return torch.stack([qx, qy, qz, qw], dim=-1)


def generate_random_quaternion_from_euler(
    batch_size: int, max_angle_deg: float, device: str, dtype: torch.dtype
) -> torch.Tensor:
    max_angle_rad = math.radians(max_angle_deg)
    roll = (2 * max_angle_rad) * torch.rand(batch_size, device=device, dtype=dtype) - max_angle_rad
    pitch = (2 * max_angle_rad) * torch.rand(batch_size, device=device, dtype=dtype) - max_angle_rad
    yaw = (2 * max_angle_rad) * torch.rand(batch_size, device=device, dtype=dtype) - max_angle_rad
    return euler_to_quaternion(roll, pitch, yaw)


def parse_floats_csv(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [float(p) for p in parts]


@dataclass
class LossParts:
    physics: torch.Tensor  # [B]
    max_joint: torch.Tensor  # [B]
    total: torch.Tensor  # [B]


def compute_loss_parts(
    physics: PhysicsLayer,
    waypoints_flat: torch.Tensor,  # [B, W*n_q]
    q0_start: torch.Tensor,  # [B, 4]
    q0_goal: torch.Tensor,  # [B, 4]
    num_waypoints: int,
    n_q: int,
    max_joint_weight: float,
) -> LossParts:
    """
    Match the training objective:
      total = physics_loss + max_joint_weight * max_joint_penalty

    - physics_loss is per-sample (not reduced) here.
    - max_joint_penalty is per-sample max(abs(joint angle)) over all waypoints/joints.
    """
    q_traj, q_dot_traj = physics.generate_trajectory(waypoints_flat)
    batch_sim_fn = torch.func.vmap(physics.simulate_single, in_dims=(0, 0, 0, 0))
    physics_loss_per, _ = batch_sim_fn(q_traj, q_dot_traj, q0_start, q0_goal)  # [B]

    waypoints_reshaped = waypoints_flat.view(-1, num_waypoints, n_q)
    max_joint_per = waypoints_reshaped.abs().view(waypoints_reshaped.size(0), -1).max(dim=1)[0]  # [B]

    total = physics_loss_per + float(max_joint_weight) * max_joint_per
    return LossParts(physics=physics_loss_per, max_joint=max_joint_per, total=total)


def summarize_tensor(x: torch.Tensor) -> Dict[str, float]:
    x = x.detach().flatten().to(torch.float32)
    finite = torch.isfinite(x)
    if finite.any():
        xf = x[finite]
        return {
            "mean": float(xf.mean().item()),
            "std": float(xf.std(unbiased=False).item()),
            "min": float(xf.min().item()),
            "max": float(xf.max().item()),
            "p50": float(xf.quantile(0.50).item()),
            "p90": float(xf.quantile(0.90).item()),
            "finite_frac": float(finite.float().mean().item()),
        }
    return {
        "mean": float("nan"),
        "std": float("nan"),
        "min": float("nan"),
        "max": float("nan"),
        "p50": float("nan"),
        "p90": float("nan"),
        "finite_frac": 0.0,
    }


def save_rows_csv(path: str, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception:  # pragma: no cover
        return None


def maybe_plot_noise_curve(rows: List[Dict[str, object]], out_path: str) -> None:
    plt = _try_import_matplotlib()
    if plt is None:
        return

    scales = [float(r["noise_scale"]) for r in rows]
    loss_mean = [float(r["total_loss_mean"]) for r in rows]
    loss_std = [float(r["total_loss_std"]) for r in rows]
    wp_l2_mean = [float(r["wp_l2_mean"]) for r in rows]
    wp_l2_std = [float(r["wp_l2_std"]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(scales, loss_mean, marker="o", label="Total loss (mean)")
    ax1.fill_between(
        scales,
        [m - s for m, s in zip(loss_mean, loss_std)],
        [m + s for m, s in zip(loss_mean, loss_std)],
        alpha=0.2,
    )
    ax1.set_xlabel("latent noise scale (sigma)")
    ax1.set_ylabel("loss")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(scales, wp_l2_mean, marker="s", color="tab:orange", label="Waypoint L2 diff (mean)")
    ax2.fill_between(
        scales,
        [m - s for m, s in zip(wp_l2_mean, wp_l2_std)],
        [m + s for m, s in zip(wp_l2_mean, wp_l2_std)],
        alpha=0.2,
        color="tab:orange",
    )
    ax2.set_ylabel("||wp(z) - wp(z0)||2")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def maybe_plot_noise_distributions(
    sample_rows: List[Dict[str, object]],
    out_path_total_loss: str,
    out_path_wp_l2: str,
) -> None:
    """
    Make sigma-wise distribution plots from per-sample rows.
    Writes two pngs:
      - total loss distribution
      - waypoint l2 distribution
    """
    plt = _try_import_matplotlib()
    if plt is None:
        return

    # group by sigma
    by_sigma: Dict[float, Dict[str, List[float]]] = {}
    for r in sample_rows:
        s = float(r["noise_scale"])
        by_sigma.setdefault(s, {"total_loss": [], "wp_l2": []})
        by_sigma[s]["total_loss"].append(float(r["total_loss"]))
        by_sigma[s]["wp_l2"].append(float(r["wp_l2"]))

    sigmas = sorted(by_sigma.keys())
    data_total = [by_sigma[s]["total_loss"] for s in sigmas]
    data_wp = [by_sigma[s]["wp_l2"] for s in sigmas]

    def _boxplot(data: List[List[float]], ylabel: str, out_path: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(list(range(1, len(sigmas) + 1)))
        ax.set_xticklabels([str(s) for s in sigmas], rotation=0)
        ax.set_xlabel("latent noise scale (sigma)")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    _boxplot(data_total, "total loss", out_path_total_loss)
    _boxplot(data_wp, "||wp(z) - wp(z0)||2", out_path_wp_l2)


def maybe_plot_dim_sweep(rows: List[Dict[str, object]], out_path: str, dim_idx: int) -> None:
    plt = _try_import_matplotlib()
    if plt is None:
        return

    xs = [float(r["z_value"]) for r in rows]
    loss = [float(r["total_loss"]) for r in rows]
    wp = [float(r["wp_l2"]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(xs, loss, marker="o", label="Total loss")
    ax1.set_xlabel(f"z[{dim_idx}]")
    ax1.set_ylabel("loss")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(xs, wp, marker="s", color="tab:orange", label="Waypoint L2 diff")
    ax2.set_ylabel("||wp - wp0||2")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_noise_scale_test(
    *,
    model: CVAE,
    physics: PhysicsLayer,
    robot_n_q: int,
    num_waypoints: int,
    latent_dim: int,
    max_joint_weight: float,
    batch_size: int,
    num_noise_samples: int,
    noise_scales: List[float],
    max_goal_angle_deg: float,
    seed: int,
    device: str,
    dtype: torch.dtype,
    out_dir: str,
    save_samples: bool,
    make_plots: bool,
) -> None:
    torch.manual_seed(seed)

    q0_start = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=dtype).repeat(batch_size, 1)
    q0_goal = generate_random_quaternion_from_euler(batch_size, max_goal_angle_deg, device=device, dtype=dtype)
    cond = torch.cat([q0_start, q0_goal], dim=1)  # [B, 8]

    # Baseline z and waypoint
    z0 = torch.randn(batch_size, latent_dim, device=device, dtype=dtype)
    with torch.no_grad():
        wp0 = model.decode(cond, z0)  # [B, W*n_q]

    rows: List[Dict[str, object]] = []
    sample_rows: List[Dict[str, object]] = []
    for sigma in noise_scales:
        # Build candidates: repeat each condition K times
        B = batch_size
        K = num_noise_samples
        cond_rep = cond.repeat_interleave(K, dim=0)  # [B*K, 8]
        q0_start_rep = q0_start.repeat_interleave(K, dim=0)
        q0_goal_rep = q0_goal.repeat_interleave(K, dim=0)
        z0_rep = z0.repeat_interleave(K, dim=0)
        eps = torch.randn(B * K, latent_dim, device=device, dtype=dtype)
        z = z0_rep + float(sigma) * eps

        with torch.no_grad():
            wp = model.decode(cond_rep, z)  # [B*K, W*n_q]
            losses = compute_loss_parts(
                physics,
                wp,
                q0_start_rep,
                q0_goal_rep,
                num_waypoints=num_waypoints,
                n_q=robot_n_q,
                max_joint_weight=max_joint_weight,
            )

        # waypoint deviation relative to per-sample baseline (repeat baseline too)
        wp0_rep = wp0.repeat_interleave(K, dim=0)
        wp_diff = wp - wp0_rep
        wp_l2 = torch.linalg.norm(wp_diff, dim=1)  # [B*K]
        wp_linf = wp_diff.abs().max(dim=1)[0]  # [B*K]

        if save_samples:
            # Save per-sample data for plotting/analysis
            for i in range(B * K):
                sample_rows.append(
                    {
                        "noise_scale": float(sigma),
                        "idx": int(i),
                        "physics_loss": float(losses.physics[i].item()),
                        "max_joint": float(losses.max_joint[i].item()),
                        "total_loss": float(losses.total[i].item()),
                        "wp_l2": float(wp_l2[i].item()),
                        "wp_linf": float(wp_linf[i].item()),
                    }
                )

        phys_s = summarize_tensor(losses.physics)
        maxj_s = summarize_tensor(losses.max_joint)
        total_s = summarize_tensor(losses.total)
        wp_l2_s = summarize_tensor(wp_l2)
        wp_linf_s = summarize_tensor(wp_linf)

        rows.append(
            {
                "noise_scale": float(sigma),
                "batch_size": int(batch_size),
                "num_noise_samples": int(num_noise_samples),
                "total_loss_mean": total_s["mean"],
                "total_loss_std": total_s["std"],
                "total_loss_p50": total_s["p50"],
                "total_loss_p90": total_s["p90"],
                "physics_loss_mean": phys_s["mean"],
                "physics_loss_std": phys_s["std"],
                "max_joint_mean": maxj_s["mean"],
                "max_joint_std": maxj_s["std"],
                "wp_l2_mean": wp_l2_s["mean"],
                "wp_l2_std": wp_l2_s["std"],
                "wp_linf_mean": wp_linf_s["mean"],
                "wp_linf_std": wp_linf_s["std"],
                "finite_frac_total_loss": total_s["finite_frac"],
            }
        )

        print(
            f"[sigma={sigma:g}] total_loss mean={total_s['mean']:.6f}, std={total_s['std']:.6f} | "
            f"wp_l2 mean={wp_l2_s['mean']:.6f}"
        )

    csv_path = os.path.join(out_dir, "noise_scale_summary.csv")
    fieldnames = list(rows[0].keys()) if rows else []
    if fieldnames:
        save_rows_csv(csv_path, fieldnames, rows)
        if make_plots:
            maybe_plot_noise_curve(rows, os.path.join(out_dir, "noise_scale_curve.png"))
        print(f"Saved: {csv_path}")

    if save_samples and sample_rows:
        samples_csv = os.path.join(out_dir, "noise_scale_samples.csv")
        save_rows_csv(samples_csv, list(sample_rows[0].keys()), sample_rows)
        print(f"Saved: {samples_csv}")
        if make_plots:
            maybe_plot_noise_distributions(
                sample_rows,
                os.path.join(out_dir, "noise_scale_box_total_loss.png"),
                os.path.join(out_dir, "noise_scale_box_wp_l2.png"),
            )


def run_dim_sweep_test(
    *,
    model: CVAE,
    physics: PhysicsLayer,
    robot_n_q: int,
    num_waypoints: int,
    latent_dim: int,
    max_joint_weight: float,
    sweep_dim: int,
    sweep_min: float,
    sweep_max: float,
    sweep_steps: int,
    max_goal_angle_deg: float,
    seed: int,
    device: str,
    dtype: torch.dtype,
    out_dir: str,
) -> None:
    torch.manual_seed(seed)
    if sweep_dim < 0 or sweep_dim >= latent_dim:
        raise ValueError(f"--sweep-dim must be in [0, {latent_dim-1}], got {sweep_dim}")

    # Single condition (B=1) to see clean per-dim effect
    q0_start = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=dtype)
    q0_goal = generate_random_quaternion_from_euler(1, max_goal_angle_deg, device=device, dtype=dtype)
    cond = torch.cat([q0_start, q0_goal], dim=1)  # [1, 8]

    z0 = torch.randn(1, latent_dim, device=device, dtype=dtype)
    with torch.no_grad():
        wp0 = model.decode(cond, z0)  # [1, W*n_q]

    z_values = torch.linspace(float(sweep_min), float(sweep_max), int(sweep_steps), device=device, dtype=dtype)
    rows: List[Dict[str, object]] = []

    for zv in z_values:
        z = z0.clone()
        z[0, sweep_dim] = zv
        with torch.no_grad():
            wp = model.decode(cond, z)
            losses = compute_loss_parts(
                physics,
                wp,
                q0_start,
                q0_goal,
                num_waypoints=num_waypoints,
                n_q=robot_n_q,
                max_joint_weight=max_joint_weight,
            )
        wp_diff = wp - wp0
        wp_l2 = torch.linalg.norm(wp_diff, dim=1)[0]
        wp_linf = wp_diff.abs().max(dim=1)[0][0]

        rows.append(
            {
                "dim": int(sweep_dim),
                "z_value": float(zv.item()),
                "physics_loss": float(losses.physics[0].item()),
                "max_joint": float(losses.max_joint[0].item()),
                "total_loss": float(losses.total[0].item()),
                "wp_l2": float(wp_l2.item()),
                "wp_linf": float(wp_linf.item()),
            }
        )

    csv_path = os.path.join(out_dir, f"dim_sweep_dim{sweep_dim}.csv")
    fieldnames = list(rows[0].keys()) if rows else []
    if fieldnames:
        save_rows_csv(csv_path, fieldnames, rows)
        maybe_plot_dim_sweep(rows, os.path.join(out_dir, f"dim_sweep_dim{sweep_dim}.png"), sweep_dim)
        print(f"Saved: {csv_path}")


def load_cvae(
    *,
    weights_path: str,
    cond_dim: int,
    output_dim: int,
    latent_dim: int,
    joint_limits: torch.Tensor,
    device: str,
    dtype: torch.dtype,
    allow_missing: bool = False,
) -> CVAE:
    model = CVAE(cond_dim, output_dim, latent_dim, joint_limits=joint_limits).to(device=device, dtype=dtype)
    if weights_path and os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device)
        try:
            model.load_state_dict(state)
        except RuntimeError as e:
            print(f"[Warning] strict=True failed, retry strict=False\n  {e}")
            model.load_state_dict(state, strict=False)
    else:
        if not allow_missing and weights_path:
            raise FileNotFoundError(f"weights not found: {weights_path}")
    model.eval()
    return model


def main() -> None:
    p = argparse.ArgumentParser(description="Test latent-space noise effect on waypoints and physics loss")
    p.add_argument("--weights", type=str, default=os.path.join(ROOT_DIR, "outputs/weights/cvae_debug/v4.pth"))
    p.add_argument(
        "--untrained",
        action="store_true",
        help="use an untrained (randomly initialized) CVAE with the same architecture",
    )
    p.add_argument(
        "--untrained-seed",
        type=int,
        default=0,
        help="seed used for random weight initialization when --untrained is set",
    )
    p.add_argument(
        "--save-untrained-pth",
        type=str,
        default="",
        help="if set (or if --untrained is set), save the untrained model state_dict to this path",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--seed", type=int, default=0)

    # Model/physics dimensions (must match weights)
    p.add_argument("--cond-dim", type=int, default=8)
    p.add_argument("--latent-dim", type=int, default=8)
    p.add_argument("--num-waypoints", type=int, default=3)
    p.add_argument("--total-time", type=float, default=10.0)

    # Loss term
    p.add_argument("--max-joint-weight", type=float, default=0.1)

    # Condition generation
    p.add_argument("--max-goal-angle-deg", type=float, default=60.0)

    # Noise-scale test
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-noise-samples", type=int, default=8, help="K samples per condition per noise scale")
    p.add_argument(
        "--noise-scales",
        type=str,
        default="0,0.05,0.1,0.2,0.5,1.0",
        help="comma-separated sigmas for z = z0 + sigma*eps",
    )

    # Per-dimension sweep test
    p.add_argument("--run-dim-sweep", action="store_true")
    p.add_argument("--sweep-dim", type=int, default=0)
    p.add_argument("--sweep-min", type=float, default=-3.0)
    p.add_argument("--sweep-max", type=float, default=3.0)
    p.add_argument("--sweep-steps", type=int, default=41)

    # Output
    p.add_argument("--out-dir", type=str, default=os.path.join(ROOT_DIR, "outputs/results/latent_noise"))
    p.add_argument("--no-plot", action="store_true", help="disable matplotlib png outputs")
    p.add_argument(
        "--save-samples",
        action="store_true",
        help="also save per-sample metrics (noise_scale_samples.csv) for distribution plots",
    )

    args = p.parse_args()

    device = args.device
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    urdf_path = os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf")
    robot, _ = urdf2robot(urdf_path, verbose_flag=False, device=device)

    output_dim = int(args.num_waypoints) * int(robot["n_q"])

    # Build model
    if args.untrained:
        # Ensure deterministic init if requested
        torch.manual_seed(int(args.untrained_seed))
        model = load_cvae(
            weights_path="",  # skip loading
            cond_dim=int(args.cond_dim),
            output_dim=output_dim,
            latent_dim=int(args.latent_dim),
            joint_limits=robot["joint_limits"],
            device=device,
            dtype=dtype,
            allow_missing=True,
        )
        # Save random init weights to a pth so it "looks like" a normal checkpoint
        save_path = args.save_untrained_pth.strip()
        if not save_path:
            save_path = os.path.join(args.out_dir, f"untrained_seed{int(args.untrained_seed)}.pth")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"[untrained] saved random-init weights to: {save_path}")
    else:
        model = load_cvae(
            weights_path=args.weights,
            cond_dim=int(args.cond_dim),
            output_dim=output_dim,
            latent_dim=int(args.latent_dim),
            joint_limits=robot["joint_limits"],
            device=device,
            dtype=dtype,
        )

    physics = PhysicsLayer(robot, int(args.num_waypoints), float(args.total_time), device)

    os.makedirs(args.out_dir, exist_ok=True)
    meta_path = os.path.join(args.out_dir, "meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"weights={args.weights}\n")
        f.write(f"untrained={bool(args.untrained)}\n")
        if args.untrained:
            f.write(f"untrained_seed={int(args.untrained_seed)}\n")
        f.write(f"device={device}\n")
        f.write(f"dtype={args.dtype}\n")
        f.write(f"cond_dim={args.cond_dim}\n")
        f.write(f"latent_dim={args.latent_dim}\n")
        f.write(f"num_waypoints={args.num_waypoints}\n")
        f.write(f"total_time={args.total_time}\n")
        f.write(f"max_joint_weight={args.max_joint_weight}\n")
        f.write(f"max_goal_angle_deg={args.max_goal_angle_deg}\n")
        f.write(f"seed={args.seed}\n")

    noise_scales = parse_floats_csv(args.noise_scales)
    run_noise_scale_test(
        model=model,
        physics=physics,
        robot_n_q=int(robot["n_q"]),
        num_waypoints=int(args.num_waypoints),
        latent_dim=int(args.latent_dim),
        max_joint_weight=float(args.max_joint_weight),
        batch_size=int(args.batch_size),
        num_noise_samples=int(args.num_noise_samples),
        noise_scales=noise_scales,
        max_goal_angle_deg=float(args.max_goal_angle_deg),
        seed=int(args.seed),
        device=device,
        dtype=dtype,
        out_dir=args.out_dir,
        save_samples=bool(args.save_samples),
        make_plots=not bool(args.no_plot),
    )

    if args.run_dim_sweep:
        run_dim_sweep_test(
            model=model,
            physics=physics,
            robot_n_q=int(robot["n_q"]),
            num_waypoints=int(args.num_waypoints),
            latent_dim=int(args.latent_dim),
            max_joint_weight=float(args.max_joint_weight),
            sweep_dim=int(args.sweep_dim),
            sweep_min=float(args.sweep_min),
            sweep_max=float(args.sweep_max),
            sweep_steps=int(args.sweep_steps),
            max_goal_angle_deg=float(args.max_goal_angle_deg),
            seed=int(args.seed),
            device=device,
            dtype=dtype,
            out_dir=args.out_dir,
        )

    print(f"Done. Results in: {args.out_dir}")


if __name__ == "__main__":
    main()


