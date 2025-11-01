import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. Domain and target data (2D grid)
N = 50
L = 1.0
x = torch.linspace(0, L, N)
y = torch.linspace(0, L, N)
X, Y = torch.meshgrid(x, y, indexing='ij')
XY = torch.stack([X.flatten(), Y.flatten()], dim=1)  # [N*N, 2]

# Example target: wave-like function in 2D with noise
u_true = (torch.sin(3 * torch.pi * X / L) * torch.sin(3 * torch.pi * Y / L) +
          0.3 * torch.sin(X / 6) * torch.sin(Y / 6)).flatten().unsqueeze(1)
u_true += 0.05 * torch.randn_like(u_true)

# Extract boundary values (edges)
u_left = u_true.view(N, N)[0, :]        # left
u_right = u_true.view(N, N)[-1, :]      # right
u_bottom = u_true.view(N, N)[:, 0]      # bottom
u_top = u_true.view(N, N)[:, -1]        # top

# Concatenate boundary values: shape [4*N]
b = torch.cat([u_left, u_right, u_bottom, u_top], dim=0).unsqueeze(0)  # [1, 4*N]

dx = x[1] - x[0]
dy = y[1] - y[0]

# 2. Encoder: maps boundary to latent parameters (e.g., 8-dim vector)
encoder = nn.Sequential(
    nn.Linear(4 * N, 64),
    nn.Tanh(),
    nn.Linear(64, 32),
    nn.Tanh(),
    nn.Linear(32, 8)  # latent code z
)

# 3. Latent forcing net: now takes (x, y, z) → f(x,y)
class LatentForcing(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + 8, 32),  # (x, y, z)
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    
    def forward(self, xy, z):
        # xy: [N*N, 2], z: [8] → expand z to [N*N, 8]
        z_expanded = z.unsqueeze(0).expand(xy.size(0), -1)
        inp = torch.cat([xy, z_expanded], dim=1)
        return self.net(inp)

forcing_net = LatentForcing()

# 4. Decoder: maps (latent u, x, y) → physical u
# We assume latent u is solved on grid, and we decode using coordinates
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + 2, 32),  # (u_latent, x, y)
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    
    def forward(self, u_latent, xy):
        inp = torch.cat([u_latent, xy], dim=1)
        return self.net(inp)

decoder = Decoder()

# 5. 2D Elliptic Solver (Jacobi iteration with zero Dirichlet BC for latent field)
def solve_elliptic_2d(f, N, dx, dy, iters=500):
    u = torch.zeros(N, N, device=f.device, dtype=f.dtype)
    f_grid = f.view(N, N)
    for _ in range(iters):
        u_old = u.clone()
        u[1:-1, 1:-1] = 0.25 * (u_old[2:, 1:-1] + u_old[:-2, 1:-1] +
                                u_old[1:-1, 2:] + u_old[1:-1, :-2] -
                                (dx**2 * dy**2) / (2 * (dx**2 + dy**2)) * f_grid[1:-1, 1:-1])
    return u.flatten().unsqueeze(1)

# 6. Training setup
params = list(encoder.parameters()) + list(forcing_net.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3)
loss_fn = nn.MSELoss()

# 7. Training loop
for epoch in range(1000):
    optimizer.zero_grad()
    
    # Encode boundary to latent code z
    z = encoder(b)  # [1, 8]
    
    # Generate latent forcing f(x,y; z)
    f = forcing_net(XY, z.squeeze(0))  # [N*N, 1]
    
    # Solve latent elliptic PDE: ∇²u = f
    u_latent = solve_elliptic_2d(f, N, dx, dy)  # [N*N, 1]
    
    # Decode to physical space using latent field + coordinates
    u_pred = decoder(u_latent, XY)  # [N*N, 1]
    
    # Loss
    loss = loss_fn(u_pred, u_true)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# 8. Plot
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X.numpy(), Y.numpy(), u_true.view(N, N).numpy(), cmap='viridis')
ax1.set_title('True')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X.numpy(), Y.numpy(), u_pred.detach().view(N, N).numpy(), cmap='viridis')
ax2.set_title(f'Predicted (Loss: {loss.item():.4f})')
plt.show()
