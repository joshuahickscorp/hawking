# Q0 clean-container for Ramanujan Tier-3 proof replay.
# All pins are immutable identifiers (commit / version+sha256 / image digest).
# Never FROM ...:latest. Never a Mathlib branch name.
#
# Build (see build.sh): network allowed to fetch pinned artifacts.
# Replay (see replay_capsule.sh): --network=none, no host state.

FROM ubuntu@sha256:4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90

LABEL org.opencontainers.image.title="ramanujan-clean-replay" \
      org.opencontainers.image.description="Pinned Lean+Mathlib+solvers for offline Tier-3 proof capsule replay" \
      ramanujan.mathlib.commit="2ec0166b31100827cd34bacca4d3b9ea3da9d618" \
      ramanujan.lean.commit="62eed1db4d67327ec8120be05f1a1b0847d74561" \
      ramanujan.lean.toolchain="leanprover/lean4:v4.33.0-rc1"

ENV DEBIAN_FRONTEND=noninteractive \
    ELAN_HOME=/opt/elan \
    PATH=/opt/elan/bin:/usr/local/bin:/usr/bin:/bin \
    MATHLIB_DIR=/opt/mathlib4 \
    LANG=C.UTF-8

# Build deps for cadical, PARI, and Lean toolchain acquisition.
# apt package versions float; the contract pins the *source/release* artifacts below.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      git \
      build-essential \
      cmake \
      unzip \
      xz-utils \
      python3 \
      libgmp-dev \
      libreadline-dev \
      zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# elan 4.2.3 (sha256-pinned linux aarch64)
# ---------------------------------------------------------------------------
ARG ELAN_URL=https://github.com/leanprover/elan/releases/download/v4.2.3/elan-aarch64-unknown-linux-gnu.tar.gz
ARG ELAN_SHA256=cb69af0803b04157bc30201c29c12fca882bb3ad8b43476b8d2d3064810bc3ac
RUN curl -fsSL -o /tmp/elan.tar.gz "$ELAN_URL" \
 && echo "${ELAN_SHA256}  /tmp/elan.tar.gz" | sha256sum -c - \
 && mkdir -p /tmp/elan-extract \
 && tar -xzf /tmp/elan.tar.gz -C /tmp/elan-extract \
 && /tmp/elan-extract/elan-init -y --default-toolchain none --no-modify-path \
 && rm -rf /tmp/elan.tar.gz /tmp/elan-extract \
 && elan --version

# ---------------------------------------------------------------------------
# Lean toolchains
#   - 4.33.0-rc1: Mathlib-required (authoritative for capsule checks)
#   - 4.32.1:     host default recorded in the environment lock
# ---------------------------------------------------------------------------
ARG LEAN_MATHLIB_TOOLCHAIN=leanprover/lean4:v4.33.0-rc1
ARG LEAN_MATHLIB_COMMIT=62eed1db4d67327ec8120be05f1a1b0847d74561
ARG LEAN_HOST_TOOLCHAIN=leanprover/lean4:v4.32.1
ARG LEAN_HOST_COMMIT=f054605aea4b840552cca2e725580bffd1e1b704

RUN elan toolchain install "$LEAN_MATHLIB_TOOLCHAIN" \
 && elan default "$LEAN_MATHLIB_TOOLCHAIN" \
 && lean --version \
 && lean --version | grep -F "$LEAN_MATHLIB_COMMIT" \
 && elan toolchain install "$LEAN_HOST_TOOLCHAIN" \
 && lean +"$LEAN_HOST_TOOLCHAIN" --version | grep -F "$LEAN_HOST_COMMIT"

# ---------------------------------------------------------------------------
# Mathlib at exact commit + lake cache (prebuilt oleans)
# ---------------------------------------------------------------------------
ARG MATHLIB_COMMIT=2ec0166b31100827cd34bacca4d3b9ea3da9d618
COPY probe/RamanujanProbe.lean /opt/ramanujan/RamanujanProbe.lean
RUN git clone --filter=blob:none https://github.com/leanprover-community/mathlib4.git "$MATHLIB_DIR" \
 && git -C "$MATHLIB_DIR" checkout --detach "$MATHLIB_COMMIT" \
 && test "$(git -C "$MATHLIB_DIR" rev-parse HEAD)" = "$MATHLIB_COMMIT" \
 && test "$(tr -d '\n' < "$MATHLIB_DIR/lean-toolchain")" = "leanprover/lean4:v4.33.0-rc1" \
 && cd "$MATHLIB_DIR" \
 && lake --version \
 && lake exe cache get \
 && lake env lean /opt/ramanujan/RamanujanProbe.lean

# ---------------------------------------------------------------------------
# Solvers: z3 4.16.0, cadical 3.0.1, PARI/GP 2.17.4 (release + sha256)
# ---------------------------------------------------------------------------
ARG Z3_URL=https://github.com/Z3Prover/z3/releases/download/z3-4.16.0/z3-4.16.0-arm64-glibc-2.38.zip
ARG Z3_SHA256=87fcd963d3eecb0f12cf1c3ef0ad74e84a3a7bd3caed5d94445645ef94ae6274
RUN curl -fsSL -o /tmp/z3.zip "$Z3_URL" \
 && echo "${Z3_SHA256}  /tmp/z3.zip" | sha256sum -c - \
 && unzip -q /tmp/z3.zip -d /tmp/z3 \
 && Z3_BIN=$(find /tmp/z3 -type f -name z3 | head -1) \
 && install -m 0755 "$Z3_BIN" /usr/local/bin/z3 \
 && rm -rf /tmp/z3 /tmp/z3.zip \
 && z3 --version | grep -F "4.16.0"

ARG CADICAL_URL=https://github.com/arminbiere/cadical/archive/refs/tags/rel-3.0.1.tar.gz
ARG CADICAL_SHA256=0a8ea563b5a25f5aa064634814edab45cc0e45111ea0f5d412a565f806fd7e11
RUN curl -fsSL -o /tmp/cadical.tar.gz "$CADICAL_URL" \
 && echo "${CADICAL_SHA256}  /tmp/cadical.tar.gz" | sha256sum -c - \
 && mkdir -p /tmp/cadical && tar -xzf /tmp/cadical.tar.gz -C /tmp/cadical --strip-components=1 \
 && cd /tmp/cadical \
 && ./configure \
 && make -j"$(nproc)" \
 && install -m 0755 build/cadical /usr/local/bin/cadical \
 && rm -rf /tmp/cadical /tmp/cadical.tar.gz \
 && cadical --version | head -1 | grep -F "3.0.1"

ARG PARI_URL=https://pari.math.u-bordeaux.fr/pub/pari/unix/pari-2.17.4.tar.gz
ARG PARI_SHA256=02651d99c391007d384b3fadbc20abc6916b77036f9e496c99e9ce8688ca4b53
RUN curl -fsSL -o /tmp/pari.tar.gz "$PARI_URL" \
 && echo "${PARI_SHA256}  /tmp/pari.tar.gz" | sha256sum -c - \
 && mkdir -p /tmp/pari && tar -xzf /tmp/pari.tar.gz -C /tmp/pari --strip-components=1 \
 && cd /tmp/pari \
 && ./Configure --prefix=/usr/local \
 && make -j"$(nproc)" gp \
 && make install \
 && rm -rf /tmp/pari /tmp/pari.tar.gz \
 && gp --version 2>&1 | head -1 | grep -F "2.17.4"

# Capsule replay entrypoint + pin manifest
COPY scripts/container_replay.sh /usr/local/bin/container_replay
COPY pins.json /opt/ramanujan/pins.json
COPY capsules/two_plus_two.capsule.json /opt/ramanujan/capsules/two_plus_two.capsule.json
RUN chmod 0755 /usr/local/bin/container_replay \
 && mkdir -p /capsule /work

WORKDIR /work
ENTRYPOINT ["/usr/local/bin/container_replay"]
CMD ["/opt/ramanujan/capsules/two_plus_two.capsule.json"]
