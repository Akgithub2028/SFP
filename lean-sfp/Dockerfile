FROM ubuntu:24.04

RUN apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*

# Install elan (Lean version manager)
RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | \
    bash -s -- --default-toolchain none -y
ENV PATH="/root/.elan/bin:${PATH}"

WORKDIR /lean-sfp
COPY lean-toolchain .
# Pre-install the toolchain
RUN elan toolchain install $(cat lean-toolchain)

COPY lakefile.lean .
# Resolve dependencies and fetch Mathlib cache
RUN lake update
RUN lake exe cache get

# Copy source files
COPY SFP.lean .
COPY SFP/ SFP/

ENTRYPOINT ["lake"]
CMD ["build"]
