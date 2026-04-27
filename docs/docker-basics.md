# Docker Basics for This Project

This guide explains the minimum commands you need for Sprint 0.

## 1) Install Docker Desktop

- Install Docker Desktop for Windows.
- Start Docker Desktop and wait until it reports it is running.

## 2) Build and run the train container

From repository root:

```bash
docker compose -f docker/compose.yml up --build train
```

Expected output includes:

`Train environment ready. model_family=imitation_baseline`

## 3) Build and run the runtime container

From repository root:

```bash
docker compose -f docker/compose.yml up --build runtime
```

Expected output includes runtime tick logs and `runtime_finished`.

## 4) Run runtime locally (without Docker)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.runtime
```

## 5) Useful notes

- Runtime config is in `configs/runtime.yaml`.
- Train config is in `configs/train.yaml`.
- Current runtime loop is a Sprint 0 scaffold with mock candidate actions.

## 6) Troubleshooting: unstable package downloads in Docker

If image build fails on `pip install` with timeout or SSL errors for `pypi.org`, the issue is usually temporary network instability from inside Docker.

This project Dockerfiles include:

- `PIP_DEFAULT_TIMEOUT=120` to avoid short default read timeouts.
- `pip install --retries 10 -r requirements.txt` to recover from transient connection drops.

If build still fails, retry the command and check VPN/proxy/firewall settings in Docker Desktop.
