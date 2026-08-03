# Doctavox API on Oracle Cloud (Always Free)

Runs the full stack — Redis + API + RQ worker + Caddy/HTTPS — on a single
**Ampere A1 Flex** VM. Always Free gives you up to **4 OCPU / 24 GB RAM**
forever, which is the only free tier with enough memory for real Demucs stem
separation and no sleep-on-idle.

The VM is ARM64. That's why the stack builds `infra/Dockerfile.slim`
(CPU-only torch) rather than `Dockerfile.api` — the `nvidia-*` pins in
`requirements.txt` have no aarch64 wheels and would fail to install.

---

## 1. Create the instance (OCI Console)

**Compute → Instances → Create instance**

| Field | Value |
|---|---|
| Image | Canonical **Ubuntu 22.04** |
| Shape | **VM.Standard.A1.Flex** (Ampere) — under "Ampere" tab |
| OCPUs / memory | **4 OCPU / 24 GB** (the whole Always Free A1 allowance) |
| Boot volume | 50 GB default is enough (~8 GB image + model cache) |
| SSH key | paste your `~/.ssh/id_ed25519.pub` |
| Public IPv4 | **Assign a public IPv4 address** — required |

If creation fails with **"Out of host capacity"**, that region has no free
Ampere left. Retry in a few hours or create the instance in another region
(the free allowance is per tenancy, and home region can't be changed — but
you can create instances in any subscribed region).

Default login user is `ubuntu`.

## 2. Open the ports (OCI security list)

**Networking → Virtual cloud networks → your VCN → Subnets → your subnet →
Security lists → Default security list → Add ingress rules**

| Source CIDR | Protocol | Destination port | Why |
|---|---|---|---|
| `0.0.0.0/0` | TCP | 80 | Let's Encrypt HTTP-01 challenge |
| `0.0.0.0/0` | TCP | 443 | the API itself |

Port 22 is already open from the default rules. Port 8000 stays closed — the
API binds to loopback only and is reached through Caddy.

> The VM *also* has its own iptables chain that rejects everything but SSH.
> Opening the security list alone is not enough; `bootstrap.sh` fixes the
> iptables half. This trips up almost everyone deploying on Oracle.

## 3. Deploy

```bash
ssh ubuntu@<PUBLIC_IP>

git clone --depth 1 https://github.com/anubhavmalhotra0395/Resume-2026.git ~/doctavox
GROQ_API_KEY=<your-key> \
APP_CORS_ORIGINS=https://doctavox.vercel.app \
  bash ~/doctavox/infra/oracle/bootstrap.sh
```

`bootstrap.sh` installs Docker, opens the host firewall, writes `.env`,
builds, starts the stack, and waits for both the API and the certificate.
The first build pulls PyTorch and takes **10–20 minutes**; later runs are
cached and take seconds.

It prints the live URL on success:

```text
Deployed: https://141-148-2-7.sslip.io
```

### About the hostname

No domain needed. [sslip.io](https://sslip.io) resolves any hostname of the
form `141-148-2-7.sslip.io` to the IP embedded in it, which lets Caddy get a
real Let's Encrypt certificate for it. Swap `API_DOMAIN` in `.env` for your
own domain later — point an A record at the VM and re-run `docker compose up
-d`, and Caddy re-issues automatically.

## 4. Point the frontend at it

The Doctavox UI is not tracked in this repo (it deploys to Vercel by CLI from
a local `vocalforge/` folder). In its `index.html`, set:

```js
const DEFAULT_REMOTE_API = "https://141-148-2-7.sslip.io";
```

Then redeploy the Vercel project. You can also test without redeploying by
appending `?api=https://141-148-2-7.sslip.io` to the frontend URL — the
standalone UI persists that override in localStorage.

Keep `APP_CORS_ORIGINS` set to your Vercel origin in production rather than
`*`.

## 5. Redeploy after a push

Re-running bootstrap is the update path — it resets the checkout to
`origin/main`, rebuilds and restarts, keeping the secrets already in `.env`:

```bash
ssh ubuntu@<PUBLIC_IP> 'bash ~/doctavox/infra/oracle/bootstrap.sh'
```

Or by hand:

```bash
cd ~/doctavox && git pull && cd infra/oracle && docker compose up -d --build
```

## Operating it

```bash
cd ~/doctavox/infra/oracle

docker compose ps                  # what's running
docker compose logs -f api         # API logs
docker compose logs -f worker      # job processing
docker compose logs -f caddy       # TLS / certificate issuance
docker compose restart worker      # after a stuck job
docker compose down                # stop (volumes survive)
curl -sS http://127.0.0.1:8000/healthz
```

Storage lives in the `app-data` volume (uploads and rendered output, cleaned
up after `APP_DELETE_AFTER_HOURS`), model weights download into it on first
job, and `caddy-data` holds the certificates — don't delete that one
casually or you'll re-request certs against Let's Encrypt's rate limit.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "Out of host capacity" on create | No free Ampere in that region. Retry later or pick another region. |
| SSH works, HTTPS times out | Security list missing 80/443, or iptables — re-run `bootstrap.sh`. |
| Caddy logs "challenge failed" | Port 80 not reachable from the internet. Both 80 and 443 must be open. |
| Browser: "blocked mixed content" | Frontend still pointing at an `http://` API. Use the HTTPS URL. |
| Jobs queue but never finish | `docker compose logs worker` — usually a model download failing. |
| Build OOMs | Confirm the shape really is 24 GB, not the 1 GB `E2.1.Micro`. |
| First job is very slow | Expected — Demucs weights download once, then cached in `app-data`. |
