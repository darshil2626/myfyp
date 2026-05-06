
# Pseudocode for Time-Dependent SINN Adapted from Current Script

This pseudocode is adapted to the structure of the current `sinn_solver_odd_time_time_in_mask.py` implementation, but replaces the **time-conditioned, slice-wise elliptic latent solve** with a **time-dependent latent PDE solve** on each sampled spatio-temporal patch.

---

## Goal

Replace the current per-time-slice latent solve

`∇·(A∇ℓ) = 0`

with a time-dependent latent PDE

`∂ℓ/∂t - ∇·(A∇ℓ) = 0`

inside each sampled cuboid patch.

---

## Main idea

For each sampled patch:

1. Build a spatial grid and time index list for that patch
2. Encode latent boundary values for every time slice
3. Encode interior latent values on the **first** time slice to define the initial condition
4. March the latent PDE forward through time within the patch
5. Compare PDE-predicted interior latents against interior encoder latents at later times
6. Decode predicted latent fields and compare with true physical fields

---

# High-level training pseudocode

```python
for epoch in range(epochs):

    sample_spatio_temporal_patches()

    epoch_losses = 0

    for patch_k in range(num_patches):

        patch_center = patch_center_idx[patch_k]
        patch_boundary_idx = patch_boundary_idx[patch_k]
        patch_interior_idx = patch_interior_idx[patch_k]
        patch_boundary_global_boundary_idx = patch_boundary_global_boundary_idx[patch_k]
        patch_boundary_global_interior_idx = patch_boundary_global_interior_idx[patch_k]

        with GradientTape() as tape:

            # 1. Encode latent values on patch boundary for all times
            latent_boundary_aligned = encode_and_align_patch_boundary_latents(
                patch_boundary_idx,
                patch_boundary_global_boundary_idx,
                patch_boundary_global_interior_idx,
                training=True
            )

            # 2. Solve time-dependent latent PDE on this patch
            total_loss, latent_loss, recon_loss, spd_loss = compute_time_dependent_pde_loss(
                patch_center_idx_tyx=patch_center,
                patch_boundary_idx_tyx=patch_boundary_idx,
                patch_interior_idx_tyx=patch_interior_idx,
                latent_boundary_aligned=latent_boundary_aligned,
                alpha_recon=1.0
            )

        grads = tape.gradient(total_loss, trainable_vars)
        grads = replace_none_with_zero(grads)
        grads = clip_gradients(grads)
        optimizer.apply_gradients(zip(grads, trainable_vars))

        accumulate_epoch_losses()

    print_epoch_summary()
```

---

# New core routine

The biggest change is to replace `compute_pde_loss(...)` with:

```python
compute_time_dependent_pde_loss(...)
```

---

## Pseudocode for `compute_time_dependent_pde_loss`

```python
def compute_time_dependent_pde_loss(
    patch_center_idx_tyx,
    patch_boundary_idx_tyx,
    patch_interior_idx_tyx,
    latent_boundary_aligned,
    alpha_recon=1.0
):
```

### Step 0: extract patch structure

```python
    patch_times = sorted(unique(patch_boundary_idx_tyx[:, 0]))

    y_min = min(patch_boundary_idx_tyx[:, 1])
    y_max = max(patch_boundary_idx_tyx[:, 1])
    x_min = min(patch_boundary_idx_tyx[:, 2])
    x_max = max(patch_boundary_idx_tyx[:, 2])

    H = y_max - y_min + 1
    W = x_max - x_min + 1
```

### Step 1: build strict spatial interior mask

```python
    spatial_interior_mask = zeros((H, W), dtype=bool)
    spatial_interior_mask[1:-1, 1:-1] = True

    interior_local_yx = argwhere(spatial_interior_mask)
    n_int = number_of_spatial_interior_points
```

### Step 2: build spatial Laplacian matrix once

```python
    K = build_2D_spatial_laplacian_on_patch_interior(H, W)
    A = get_latent_operator_matrix()
```

This corresponds to the PDE

`∂ℓ/∂t - ∇·(A∇ℓ) = 0`

where the spatial part is discretised using `K`.

---

## Option A: explicit forward Euler time marching

For time step `n -> n+1`:

`(ℓ^{n+1} - ℓ^n)/Δt = diffusion_term(ℓ^n, boundary^n)`

so

`ℓ^{n+1} = ℓ^n + Δt * diffusion_term(ℓ^n, boundary^n)`

---

### Step 3: define initial condition from first time slice

```python
    t0 = patch_times[0]

    idx_init = stack_indices_for_all_patch_interior_points(t=t0)
    feats_init = stack_mask_patch_features_from_idx(idx_init)
    latent_current = interior_encoder(feats_init, training=True)   # shape (n_int, r)
```

This gives the latent state on the first slice: `ℓ^0`.

---

### Step 4: march forward through time

```python
    latent_consistency_loss = 0.0
    reconstruction_loss = 0.0
    num_steps = 0

    for n in range(len(patch_times) - 1):

        tn = patch_times[n]
        tn1 = patch_times[n + 1]
        dt = T[tn1] - T[tn]
```

#### 4a. Extract boundary latents at current time

```python
        boundary_latents_n = gather_boundary_latents_for_time(
            patch_boundary_idx_tyx,
            latent_boundary_aligned,
            time_index=tn
        )
```

#### 4b. Build boundary RHS for the diffusion operator

```python
        rhs_boundary_n = assemble_boundary_contribution(
            boundary_latents_n,
            patch_geometry,
            A
        )
```

#### 4c. Compute diffusion term on current latent interior

```python
        diffusion_term = - (K ⊗ A) @ latent_current + rhs_boundary_n
```

Conceptually this is the discrete version of `∇·(A∇ℓ)` on the patch interior.

#### 4d. Forward Euler update

```python
        latent_next_pred = latent_current + dt * diffusion_term
```

---

### Step 5: compare with encoder latent target at next time slice

```python
        idx_target = stack_indices_for_all_patch_interior_points(t=tn1)
        feats_target = stack_mask_patch_features_from_idx(idx_target)
        latent_next_true = interior_encoder(feats_target, training=True)

        latent_consistency_loss += mean_square(latent_next_pred - latent_next_true)
```

---

### Step 6: decode predicted latent field and compare with physical field

```python
        u_pred = decoder(latent_next_pred, training=True)

        u_true = gather_true_U_values_for_patch_interior(time=tn1)

        reconstruction_loss += mean_square(u_pred - u_true)
```

---

### Step 7: roll state forward

```python
        latent_current = latent_next_pred
        num_steps += 1
```

---

### Step 8: average losses and return

```python
    latent_loss = latent_consistency_loss / num_steps
    recon_loss = reconstruction_loss / num_steps
    spd_loss = 0.0

    total_loss = latent_loss + alpha_recon * recon_loss + spd_loss

    return total_loss, latent_loss, recon_loss, spd_loss
```

---

# Cleaner matrix form

If you want something closer to the current linear-algebra style, define:

- `L_space = spatial Laplacian on patch interior`
- `A = latent SPD matrix`
- `M = L_space ⊗ A`

Then the explicit update is:

```python
latent_next_pred = latent_current + dt * (- M @ latent_current + rhs_boundary)
```

where:

- `latent_current` is flattened as shape `(n_int * r, 1)`
- `rhs_boundary` contains the spatial boundary forcing contribution

Then reshape back to `(n_int, r)` before decoding.

---

# Option B: implicit backward Euler

This is numerically more stable and closer to your current solve style.

Instead of:

`ℓ^{n+1} = ℓ^n + dt * diffusion_term`

solve:

`(ℓ^{n+1} - ℓ^n)/dt - ∇·(A∇ℓ^{n+1}) = 0`

which becomes:

`(I + dt * M) ℓ^{n+1} = ℓ^n + dt * rhs_boundary^{n+1}`

This is very attractive because it preserves the idea of solving a linear system.

---

## Pseudocode for implicit update

Inside the time loop:

```python
        system_matrix = I + dt * M
        rhs = latent_current_flat + dt * rhs_boundary_flat

        latent_next_pred_flat = solve(system_matrix, rhs)
        latent_next_pred = reshape(latent_next_pred_flat, (n_int, r))
```

This is probably the best option if you want the code to stay stylistically close to your current implementation.

---

# Recommended implementation route

For your codebase, the easiest progression is:

## Stage 1
Keep everything else the same, but replace the per-time steady solve with:

- one initial latent interior state
- forward time marching through the patch

## Stage 2
Once working, switch from explicit Euler to backward Euler for stability.

---

# How this maps to your current script

## Current function to replace

```python
compute_pde_loss(...)
```

This currently:

- loops through time slices
- solves independent spatial elliptic problems
- compares each one to encoder latents
- decodes each one

## New function

```python
compute_time_dependent_pde_loss(...)
```

This should instead:

- build the spatial operator once
- initialise latent interior at first patch time
- evolve latent field forward in time
- compare predicted later-time latents to encoder latents
- decode later-time predictions

---

# Suggested helper functions

To keep code clean, split into helper functions:

```python
build_patch_spatial_operator(...)
assemble_boundary_rhs_for_time(...)
encode_initial_latent_state(...)
encode_target_latent_state(...)
decode_latent_state(...)
```

---

# Full compact pseudocode

```python
def compute_time_dependent_pde_loss(...):

    patch_times = sorted(unique times in patch)

    K = build_spatial_laplacian()
    A = get_latent_operator_matrix()
    M = kron(K, A)

    # initial condition from first slice
    latent_current = encode_interior_latents_at_time(t0)

    total_latent_loss = 0
    total_recon_loss = 0

    for n in range(len(patch_times) - 1):

        tn  = patch_times[n]
        tn1 = patch_times[n + 1]
        dt  = time_difference(tn, tn1)

        rhs_boundary = assemble_boundary_rhs_for_time(tn or tn1)

        # explicit version
        latent_next_pred = latent_current + dt * (-M @ latent_current + rhs_boundary)

        # OR implicit version
        # solve (I + dt*M) latent_next_pred = latent_current + dt*rhs_boundary

        latent_next_true = encode_interior_latents_at_time(tn1)
        u_pred = decode(latent_next_pred)
        u_true = true_field_at_time(tn1)

        total_latent_loss += mse(latent_next_pred, latent_next_true)
        total_recon_loss += mse(u_pred, u_true)

        latent_current = latent_next_pred

    return total_loss, latent_loss, recon_loss, spd_loss
```

---

# Interpretation

Old model:

- encoder sees time
- PDE does not

New model:

- encoder sees time
- PDE also evolves through time

This is the clean mathematical step from a **time-conditioned elliptic SINN** to a **genuinely time-dependent latent PDE SINN**.
