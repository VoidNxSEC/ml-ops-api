# GitHub Actions Workflows - ML Offload API

## Workflows

### 1. `ci.yml` - Continuous Integration

**Triggers**: Push to main/develop, Pull Requests, Manual dispatch

**Jobs**:

#### Rust CI
- **Matrix**: stable, beta
- **Steps**:
  - Code formatting check (`cargo fmt`)
  - Linting (`cargo clippy`)
  - Build (`cargo build`)
  - Tests (`cargo test`)
  - Release build
  - Upload binary artifact

#### Nix Flake CI
- **Steps**:
  - Flake check (`nix flake check`)
  - Build all packages (default, rust, python, all)
  - Test devShell

#### Security Audit
- **Steps**:
  - `cargo audit` (security vulnerabilities)
  - `cargo outdated` (dependency updates)

#### Code Coverage
- **Tool**: cargo-llvm-cov
- **Upload**: Codecov
- **Format**: LCOV

#### Debug Session
- **Trigger**: On failure OR manual enable
- **Tool**: tmate (30min timeout)
- **Access**: Actor-limited

---

### 2. `release.yml` - Release Automation

**Trigger**: Git tags `v*.*.*`, Manual dispatch

**Jobs**:

#### Create Release
- Auto-generate release notes
- Create GitHub Release

#### Build Binaries
- **Platforms**:
  - x86_64-unknown-linux-gnu (Ubuntu)
  - x86_64-apple-darwin (macOS)
- **Output**: Compressed tarballs
- **Upload**: GitHub Release assets

#### Build Nix
- Nix package build
- Bundle to Docker image
- Upload to GitHub Release

---

### 3. `docker.yml` - Container Build

**Triggers**: Push to main, Tags, Pull Requests, Manual dispatch

**Registry**: GitHub Container Registry (ghcr.io)

**Steps**:
- Build with Nix (`nix build .#dockerImage`)
- Tag with metadata (branch, semver, SHA)
- Push to ghcr.io

**Tags**:
- `main` branch → `latest`
- `v1.2.3` tag → `1.2.3`, `1.2`
- PR → `pr-123`
- Commit SHA → `sha-abc1234`

---

## Usage

### Running CI Locally

```bash
# Rust checks
cd api/
cargo fmt --check
cargo clippy --all-targets --all-features
cargo test --all-features
cargo build --release

# Nix checks
nix flake check
nix build
nix develop --command cargo test
```

### Manual Workflow Dispatch

```bash
# Trigger CI with tmate debug
gh workflow run ci.yml -f enable-tmate=true

# Trigger Docker build
gh workflow run docker.yml
```

### Creating a Release

```bash
# Tag and push
git tag v0.1.0
git push origin v0.1.0

# Release workflow will auto-trigger
# - Create GitHub Release
# - Build binaries for Linux + macOS
# - Build Nix package
# - Upload all artifacts
```

---

## Secrets Required

### Optional (for full features):

- `CODECOV_TOKEN` - For code coverage upload
- GitHub Token automatically provided for:
  - GitHub Container Registry
  - Release creation
  - Artifact uploads

---

## Cache Strategy

### Cargo Cache
- Registry: `~/.cargo/registry`
- Index: `~/.cargo/git`
- Build: `api/target`
- Key: `${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}`

### Nix Cache
- Magic Nix Cache (automatic)
- Shared across workflows
- Significantly faster builds

---

## CI Performance

### Typical Run Times:
- **Rust CI**: ~5-8 minutes (with cache)
- **Nix CI**: ~3-5 minutes (with Magic Cache)
- **Security**: ~2-3 minutes
- **Coverage**: ~6-10 minutes
- **Total**: ~10-15 minutes (parallel)

### Optimization Tips:
- Cargo caching reduces build time by 70%
- Magic Nix Cache speeds up Nix builds by 80%
- Matrix builds run in parallel
- Artifacts retained for 7 days

---

## Troubleshooting

### CI Failing?

```bash
# Enable tmate debug
gh workflow run ci.yml -f enable-tmate=true
# SSH into runner and investigate
```

### Build Issues?

```bash
# Check locally first
nix flake check --show-trace
cargo check --all-targets
```

### Cache Issues?

```bash
# Clear GitHub Actions cache
gh cache delete <cache-key>
# Or clear all
gh cache list | awk '{print $1}' | xargs -I {} gh cache delete {}
```

---

## Status Badges

Add to README.md:

```markdown
[![CI](https://github.com/VoidNxSEC/ml-offload-api/actions/workflows/ci.yml/badge.svg)](https://github.com/VoidNxSEC/ml-offload-api/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/VoidNxSEC/ml-offload-api/branch/main/graph/badge.svg)](https://codecov.io/gh/VoidNxSEC/ml-offload-api)
[![Docker](https://github.com/VoidNxSEC/ml-offload-api/actions/workflows/docker.yml/badge.svg)](https://github.com/VoidNxSEC/ml-offload-api/actions/workflows/docker.yml)
```

---

## Next Steps

- [ ] Add integration tests
- [ ] Add benchmark workflow
- [ ] Add dependency review
- [ ] Add SLSA provenance
- [ ] Add automated dependency updates (Dependabot/Renovate)
