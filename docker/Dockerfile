# The p2pgpu session image.
#
# Two problems with pulling a stock PyTorch image at share time:
#
#   1. The '-devel' tag is 7-9 GB because it carries the full CUDA toolkit
#      (nvcc, headers) that almost nobody needs -- those are for compiling
#      custom CUDA extensions, not for training. '-runtime' is roughly half.
#   2. Even after the pull, the container spent 2-3 minutes doing
#      'pip install jupyterlab' on every single start. That work is identical
#      every time, so it belongs in a layer, not in the startup path.
#
# Baking both away turns "press the button and wait 25 minutes" into "press the
# button and it is ready", which is the whole point of the app.
#
# Build and publish (maintainer only):
#   docker buildx build --platform linux/amd64 \
#     -t <yourname>/p2pgpu-session:cuda12.4-torch2.5 --push docker/

ARG BASE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${BASE}

LABEL org.opencontainers.image.title="p2pgpu session"
LABEL org.opencontainers.image.description="GPU session image: PyTorch + JupyterLab + sshd"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    JUPYTER_ALLOW_INSECURE_WRITES=1

# openssh-server is installed here rather than at container start, where it cost
# an apt-get update plus download every session.
RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends \
      openssh-server \
      git \
      curl \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /run/sshd \
 && sed -i 's/^#*PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config \
 && sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config

# Pinned so a session image is reproducible; a surprise JupyterLab major
# version breaking the notebook mid-share is not a good failure mode.
RUN pip install --no-cache-dir \
      jupyterlab==4.3.4 \
      ipywidgets==8.1.5 \
      matplotlib==3.10.0 \
      tqdm==4.67.1

WORKDIR /workspace
EXPOSE 8888 22

# The entrypoint handles the SSH key and the expiry timer; see entrypoint.sh.
COPY entrypoint.sh /usr/local/bin/p2pgpu-entrypoint
RUN chmod +x /usr/local/bin/p2pgpu-entrypoint

ENTRYPOINT ["/usr/local/bin/p2pgpu-entrypoint"]
