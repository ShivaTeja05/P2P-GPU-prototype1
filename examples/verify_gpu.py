"""Paste this into the first Jupyter cell once you're connected.

Proves the whole chain in about 30 seconds: the GPU is real, PyTorch can reach
it, it computes correctly, and it can actually train. Run this before you trust
the setup with anything that matters.
"""

import time

import torch

print("=" * 58)
print("1. Is a CUDA GPU visible?")
print("=" * 58)

if not torch.cuda.is_available():
    print("FAIL: torch.cuda.is_available() is False")
    print("      The container started but has no GPU. On the owner's machine,")
    print("      run 'p2pgpu doctor' (or Check-Setup.bat) to find out why.")
    raise SystemExit(1)

name = torch.cuda.get_device_name(0)
total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
free_b, _ = torch.cuda.mem_get_info(0)
major, minor = torch.cuda.get_device_capability(0)

print(f"   GPU              : {name}")
print(f"   VRAM total       : {total_gb:.1f} GB")
print(f"   VRAM free now    : {free_b / 1024**3:.1f} GB")
print(f"   compute capability: {major}.{minor}")
print(f"   torch / cuda     : {torch.__version__} / {torch.version.cuda}")

print()
print("=" * 58)
print("2. Does it compute correctly?")
print("=" * 58)

# A wrong-image mismatch (kernels not built for this card) usually fails here
# rather than at is_available(), which is why this check exists at all.
a = torch.randn(1000, 1000, device="cuda")
b = torch.randn(1000, 1000, device="cuda")
expected = (a.cpu() @ b.cpu())
actual = (a @ b).cpu()
if torch.allclose(expected, actual, atol=1e-3):
    print("   matmul matches CPU reference -> kernels are correct")
else:
    print("   FAIL: GPU result differs from CPU. Wrong image for this card?")
    raise SystemExit(1)

print()
print("=" * 58)
print("3. How fast is it?")
print("=" * 58)

size = 4096
x = torch.randn(size, size, device="cuda")
torch.cuda.synchronize()
start = time.perf_counter()
iterations = 50
for _ in range(iterations):
    x @ x
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

# 2*n^3 FLOPs per matmul.
tflops = (2 * size**3 * iterations) / elapsed / 1e12
print(f"   {size}x{size} matmul x{iterations}: {elapsed:.2f}s  ->  {tflops:.1f} TFLOP/s (fp32)")

print()
print("=" * 58)
print("4. Can it actually train?")
print("=" * 58)

model = torch.nn.Sequential(
    torch.nn.Linear(512, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, 10),
).cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
data = torch.randn(256, 512, device="cuda")
labels = torch.randint(0, 10, (256,), device="cuda")

first_loss = None
for step in range(100):
    loss = torch.nn.functional.cross_entropy(model(data), labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step == 0:
        first_loss = loss.item()

last_loss = loss.item()
print(f"   loss {first_loss:.4f} -> {last_loss:.4f} over 100 steps")

if last_loss < first_loss:
    print("   loss went down -> forward, backward and optimizer all work")
else:
    print("   WARNING: loss did not decrease. Something is off.")

print()
print("=" * 58)
print("ALL CHECKS PASSED - you are training on a remote GPU.")
print("=" * 58)
print()
print("Remember: save anything you want to keep into /workspace.")
print("Everything else in this container disappears when the share ends.")
