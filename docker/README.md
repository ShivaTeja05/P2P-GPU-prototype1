# The optional prebuilt session image

**You do not need this.** `p2pgpu share` pulls a stock PyTorch image and
installs JupyterLab on start, which works everywhere and needs no build step.

This image exists to remove that startup cost. It bakes JupyterLab and `sshd`
into a layer, so a share becomes "press the button and it is ready" instead of
"press the button and wait two minutes" — worth it if you share often, pointless
if you share occasionally.

## Build and publish

```bash
docker buildx build --platform linux/amd64 \
  -t youruser/p2pgpu-session:cuda12.4-torch2.5 --push docker/
```

## Use it

```bash
p2pgpu share --image youruser/p2pgpu-session:cuda12.4-torch2.5
```

Nothing else changes. `--image` bypasses the automatic image selection entirely,
so if you build this, **you** are responsible for matching it to the GPU — see
the compatibility table in the main README. The automatic path exists precisely
because that matching is easy to get wrong.

## Why it is not the default

Publishing an image means hosting it, versioning it, and keeping it current with
PyTorch and CUDA releases. A stock upstream image has none of those obligations
and is already cached on many machines. The default optimises for "a stranger
can run this today" over "a regular user saves two minutes".
