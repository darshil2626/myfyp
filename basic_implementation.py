import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. Domain and target data
N = 50
L = 1.0
x = torch.linspace(0, L, N).unsqueeze(1)
u_true = torch.sin(3*torch.pi*x/L) + 0.3*torch.sin(x/6) + 0.05*torch.randn_like(x)
b = torch.cat([u_true[0], u_true[-1]])  # boundary values [u_left, u_right]
dx = x[1] - x[0]

# 2. Encoder (maps boundary -> latent boundary)
encoder = nn.Sequential(
    nn.Linear(2, 8), nn.Tanh(),
    nn.Linear(8, 2)
)

# 3. Latent forcing (maps x -> f(x))
forcing_net = nn.Sequential(
    nn.Linear(1, 16), nn.Tanh(),
    nn.Linear(16, 16), nn.Tanh(),
    nn.Linear(16, 1)
)

# 4. Decoder (maps latent u -> physical u)
decoder = nn.Sequential(
    nn.Linear(1, 8), nn.Tanh(),
    nn.Linear(8, 1)
)

# 5. PDE solver: A u'' = f(x)
def solve_elliptic(f, b_latent, dx, iters=200):
    n = len(f)
    u = torch.zeros_like(f)
    u[0], u[-1] = b_latent[0], b_latent[1]
    for _ in range(iters):
        u[1:-1] = 0.5 * (u[:-2] + u[2:] - dx**2 * f[1:-1])
    return u

# 6. Training setup
params = list(encoder.parameters()) + list(forcing_net.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3)
loss_fn = nn.MSELoss()

# 7. Training loop
for epoch in range(1000):
    optimizer.zero_grad()
    b_latent = encoder(b.unsqueeze(0))  # shape [1,2] -> latent boundary
    f = forcing_net(x)
    u_latent = solve_elliptic(f, b_latent.squeeze(0), dx)
    u_pred = decoder(u_latent)  # decode latent to physical
    loss = loss_fn(u_pred, u_true)
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# 8. Plot
plt.plot(x.detach(), u_true.detach(), 'k', label='True')
plt.plot(x.detach(), u_pred.detach(), 'r--', label='Predicted')
plt.xlabel('x'); plt.ylabel('u(x)')
plt.legend(); plt.title(f"Loss = {loss.item():.4f}")
plt.show()
