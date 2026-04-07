"""
Source term diagnostic for v4 SINN.

Run this AFTER training by adding it to the bottom of your v4 script
(after the solver has been trained), or import the function and call it
on your trained solver object.

This measures:
  1. Mean absolute magnitude of the source term s(x,y) at each test timestep
  2. Mean absolute magnitude of the boundary RHS contribution
  3. Their ratio — if source/boundary << 1, the source is being ignored
  4. Per-latent-dimension breakdown to see if specific dimensions are dead
"""

import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from scipy.sparse import diags


def diagnose_source_magnitude(solver, n_steps=20):
    """
    Run sequential reconstruction on the first n_steps of the test set
    and measure source vs boundary contributions at each step.

    Args:
        solver: trained sinn object (from v4_source_sinn.py)
        n_steps: number of test timesteps to diagnose
    """
    b_thick = int(solver.b_thick)

    # --- Shared spatial structures (same as reconstruct_sequence) ---
    obs_mask = np.zeros((solver.ny, solver.nx), dtype=bool)
    obs_mask[:b_thick, :] = True; obs_mask[-b_thick:, :] = True
    obs_mask[:, :b_thick] = True; obs_mask[:, -b_thick:] = True

    interior_indices = np.argwhere(~obs_mask)
    boundary_indices = np.argwhere(obs_mask)
    num_interior = interior_indices.shape[0]

    interior_row_map = -np.ones((solver.ny, solver.nx), dtype=np.int32)
    for rid, (y, x) in enumerate(interior_indices):
        interior_row_map[y, x] = rid

    bnd_map = {(int(y), int(x)): i for i, (y, x) in enumerate(boundary_indices)}
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    bnd_contribs = [[] for _ in range(num_interior)]
    for rid, (y, x) in enumerate(interior_indices):
        for dy, dx in steps:
            ny_, nx_ = int(y + dy), int(x + dx)
            bi = bnd_map.get((ny_, nx_))
            if bi is not None:
                bnd_contribs[rid].append(bi)

    A_np = solver.get_latent_operator_matrix().numpy().astype(np.float64)
    A_np = 0.5 * (A_np + A_np.T)

    # --- Feature builder for boundary (observed only) ---
    def stack_observed_only(idx_tyx):
        idx_tyx = np.asarray(idx_tyx, dtype=np.int32)
        N = idx_tyx.shape[0]
        r = int(solver.mask_radius)
        win = 2 * r + 1; ws = win * win
        nts = 1 + solver.n_past_steps
        feats = np.zeros((N, 3 + 2 * ws * nts), dtype=np.float32)
        feats[:, 0] = idx_tyx[:, 0].astype(np.float32) / max(solver.nt - 1, 1)
        feats[:, 1] = idx_tyx[:, 1].astype(np.float32) / max(solver.ny - 1, 1)
        feats[:, 2] = idx_tyx[:, 2].astype(np.float32) / max(solver.nx - 1, 1)
        u_off = 3; m_off = 3 + ws * nts
        for i in range(N):
            ti, yi, xi = idx_tyx[i]
            for p in range(nts):
                ts = int(ti) - p; so = p * ws
                if ts < 0: continue
                ptr = 0
                for dy in range(-r, r + 1):
                    yy = yi + dy
                    for dx in range(-r, r + 1):
                        xx = xi + dx
                        if 0 <= yy < solver.ny and 0 <= xx < solver.nx and obs_mask[yy, xx]:
                            feats[i, u_off + so + ptr] = float(solver.U[ts, yy, xx])
                            feats[i, m_off + so + ptr] = 1.0
                        ptr += 1
        return feats

    # --- Run diagnostic ---
    last_train_t = int(solver.train_time_indices[-1])
    test_t = np.sort(solver.test_time_indices)[:n_steps]
    all_t = np.concatenate([[last_train_t], test_t])

    prev_latent = None

    # Check if solver has learnable source_scale
    has_scale = hasattr(solver, 'source_scale')
    if has_scale:
        scale_val = float(solver.source_scale.numpy())
        print(f"Learned source_scale: {scale_val:.6f}")
    else:
        scale_val = 1.0
        print("No source_scale found (raw source used)")

    print("=" * 80)
    print("SOURCE TERM DIAGNOSTIC")
    print("=" * 80)
    print(f"{'t':>5} | {'Bnd RHS |mean|':>15} | {'Raw src |mean|':>15} | "
          f"{'Eff src |mean|':>15} | {'Eff ratio s/b':>14}")
    print("-" * 80)

    # Storage for per-dimension analysis
    all_source_per_dim = []

    for i, t in enumerate(all_t):
        t = int(t)

        # Encode boundary
        bnd_y = boundary_indices[:, 0].astype(np.int32)
        bnd_x = boundary_indices[:, 1].astype(np.int32)
        bnd_t = np.full(boundary_indices.shape[0], t, dtype=np.int32)
        idx_bnd = np.stack([bnd_t, bnd_y, bnd_x], axis=1)
        bnd_feats = stack_observed_only(idx_bnd)
        bnd_latents = solver.boundary_encoder(
            tf.constant(bnd_feats, tf.float32), training=False
        ).numpy()
        bnd_latents = np.nan_to_num(bnd_latents, nan=0.0)

        # Boundary RHS
        rhs_bnd = np.zeros((num_interior, solver.num_latentdim), dtype=np.float64)
        for rid in range(num_interior):
            for bidx in bnd_contribs[rid]:
                rhs_bnd[rid, :] += A_np @ bnd_latents[bidx, :].astype(np.float64)

        bnd_mag = float(np.abs(rhs_bnd).mean())

        # Source term
        if prev_latent is not None:
            src_feats = solver._build_source_features_global(
                prev_latent, interior_indices, interior_row_map, t
            )
            src_vals = solver.source_network(
                tf.constant(src_feats, tf.float32), training=False
            ).numpy()
            src_vals = np.nan_to_num(src_vals, nan=0.0)

            src_mag = float(np.abs(src_vals).mean())
            src_std = float(np.std(src_vals))
            src_max = float(np.abs(src_vals).max())
            eff_mag = src_mag * scale_val
            eff_ratio = eff_mag / bnd_mag if bnd_mag > 1e-10 else float('inf')

            all_source_per_dim.append(np.abs(src_vals).mean(axis=0))

            label = "SEED" if i == 0 else ""
            print(f"{t:5d} | {bnd_mag:15.6f} | {src_mag:15.6f} | "
                  f"{eff_mag:15.6f} | {eff_ratio:14.4f}")
        else:
            print(f"{t:5d} | {bnd_mag:15.6f} | {'(no prev)':>15} | "
                  f"{'--':>15} | {'--':>14}  <- t=0, s=0")

        # Solve to get latent field for next step (simplified: use boundary-only solve)
        # We need the latent field, so do a quick CG solve
        eigvals_A, Q_A = np.linalg.eigh(A_np)
        if float(np.min(eigvals_A)) <= 1e-8:
            eigvals_A = eigvals_A + 1e-6

        rhs_total = rhs_bnd.copy()
        if prev_latent is not None:
            rhs_total = rhs_total + scale_val * src_vals.astype(np.float64)

        rhs_tilde = rhs_total @ Q_A

        # Quick Laplacian (rebuild — not ideal but keeps diagnostic self-contained)
        rows, cols, vals = [], [], []
        for rid, (y, x) in enumerate(interior_indices):
            rows.append(rid); cols.append(rid); vals.append(4.0)
            for dy, dx in steps:
                ny_, nx_ = int(y + dy), int(x + dx)
                nr = interior_row_map[ny_, nx_]
                if nr >= 0:
                    rows.append(rid); cols.append(int(nr)); vals.append(-1.0)

        K = sp.csr_matrix((vals, (rows, cols)), shape=(num_interior, num_interior), dtype=np.float64)
        diag_K = np.array(K.sum(axis=1)).ravel()
        diag_K[diag_K == 0] = 1.0
        M_pre = diags(1.0 / diag_K, dtype=np.float64, format='csr')

        from scipy.sparse.linalg import cg
        lat_tilde = np.zeros_like(rhs_tilde, dtype=np.float64)
        for k in range(solver.num_latentdim):
            b = rhs_tilde[:, k] / float(eigvals_A[k])
            if np.isnan(b).any():
                continue
            xk, _ = cg(K, b, M=M_pre, tol=1e-2, maxiter=500)
            lat_tilde[:, k] = xk

        prev_latent = (lat_tilde @ Q_A.T).astype(np.float32)

    # --- Per-dimension summary ---
    if len(all_source_per_dim) > 0:
        dim_means = np.mean(all_source_per_dim, axis=0)
        print("\n" + "=" * 80)
        print("PER LATENT DIMENSION: Mean |source| across all test steps")
        print("=" * 80)
        for d in range(solver.num_latentdim):
            bar = "#" * int(dim_means[d] / (dim_means.max() + 1e-10) * 40)
            print(f"  dim {d:2d}: {dim_means[d]:.6f}  {bar}")

        dead_dims = np.sum(dim_means < 0.01 * dim_means.max())
        print(f"\n  Dimensions with <1% of max activation: {dead_dims}/{solver.num_latentdim}")

        overall_src = np.mean([np.abs(s).mean() for s in all_source_per_dim])
        print(f"\n  Overall mean |source|: {overall_src:.6f}")

    print("\n" + "=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)
    print("  Ratio s/b >> 0.1  : Source is meaningfully contributing")
    print("  Ratio s/b ~ 0.01  : Source is marginal, barely affecting the solve")
    print("  Ratio s/b < 0.001 : Source is effectively dead — model ignores it")
    print("")
    print("  If source is dead, possible fixes:")
    print("    - Train longer (source learns slower than encoders)")
    print("    - Increase source network capacity (more units/layers)")
    print("    - Add a weighting term to the loss that rewards non-zero source")
    print("    - Pre-train source network on latent field differences")
    print("=" * 80)


# --- Run if called directly or appended to training script ---
if __name__ == "__main__":
    print("This script should be called after training.")
    print("Add the following to the end of your v4 training script:")
    print("")
    print("  from diagnose_source import diagnose_source_magnitude")
    print("  diagnose_source_magnitude(solver, n_steps=20)")