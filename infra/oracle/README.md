# Deploy the API on Oracle Cloud (Always Free)

This path avoids Fly.io’s card requirement: you run **Docker Compose** on an **Oracle Ampere A1 Flex** VM (ARM64). The stack is **Redis + API** (RQ worker starts inside the API container when `APP_ENABLE_WORKER=1`).

**Important:** Oracle’s free Ampere VMs are **ARM (aarch64)**. The `Dockerfile.api` build uses multi-arch Python images; PyTorch publishes **linux/arm64** wheels. If a dependency fails to build on ARM, try a smaller x86 shape (see Oracle docs) or trim optional ML packages.

## 1. Oracle Cloud account and region

1. Sign up at [Oracle Cloud Free Tier](https://www.oracle.com/cloud/free/).
2. Create a tenancy and sign in to the **Console**.
3. Pick a **region** where **Ampere A1** capacity exists (e.g. Ashburn, Phoenix, Frankfurt). If instance creation fails with “out of capacity”, switch region or retry later.

## 2. Networking (VCN)

1. **Networking → Virtual cloud networks** → create VCN (Create VCN wizard with public subnet is fine).
2. Open the subnet’s **Security list** (or NSG) and add **ingress**:
   - **SSH:** TCP **22** from **your IP** only (recommended), not the whole internet.
   - **API (testing):** TCP **8000** from `0.0.0.0/0` (or restrict to your IP while testing).
3. Later, for HTTPS on **443**, either open **443** here and use **Caddy** on the host (`Caddyfile.example`), or put **8000** behind Caddy only and drop public 8000.

## 3. Compute instance (Ampere A1 Flex)

1. **Compute → Instances → Create instance**.
2. **Image:** Canonical **Ubuntu 22.04** (aarch64).
3. **Shape:** **VM.Standard.A1.Flex** (Ampere).
4. **OCPUs / memory:** Always Free allows up to **4 OCPUs and 24 GB** total **per tenancy** for A1 Flex (see current [Oracle Free Tier](https://docs.oracle.com/en-us/iaas/Content/FreeTier/resourceref.htm) limits). A practical start is **2 OCPUs + 12 GB** for this API stack.
5. **SSH key:** paste your **public** key (`id_ed25519.pub` or `id_rsa.pub`).
6. **Public IPv4:** assign a public IP so you can SSH and reach the API.
7. Create the instance and wait until it is **Running**.

## 4. SSH into the VM

```bash
ssh -i /path/to/your.key ubuntu@<PUBLIC_IP>
```

(Username may be `ubuntu` or `opc` depending on image; Oracle shows the default user in instance details.)

## 5. Install Docker (Ubuntu ARM)

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl git
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "${VERSION_CODENAME:-jammy}") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker "$USER"
```

Log out and SSH back in so the `docker` group applies.

## 6. Get the application on the VM

**Option A — Git clone (if the repo is on GitHub):**

```bash
git clone https://github.com/YOUR_USER/YOUR_REPO.git
cd YOUR_REPO/infra/oracle
```

**Option B — Copy from your PC** (PowerShell example):

```powershell
scp -i path\to\key -r "C:\Users\...\Desktop\doc" ubuntu@PUBLIC_IP:~/
```

Then on the VM:

```bash
cd ~/doc/infra/oracle
```

## 7. Configure and start

```bash
cp .env.example .env
nano .env   # set APP_CORS_ORIGINS (comma-separated origins, e.g. https://x.vercel.app); add optional secrets
docker compose up -d --build
```

First **build can take a long time** (PyTorch, demucs, etc.). Watch logs:

```bash
docker compose logs -f api
```

Smoke test:

```bash
curl -sS http://127.0.0.1:8000/healthz
```

From your laptop (security list must allow 8000):

```text
http://<PUBLIC_IP>:8000/healthz
```

## 8. Point Vercel at Oracle

Set your frontend’s API base URL to:

```text
http://<PUBLIC_IP>:8000
```

For production, use **HTTPS** (Caddy + domain) and set `APP_CORS_ORIGINS` to your Vercel origin (comma-separated if you have several).

## 9. Optional: HTTPS with Caddy

1. Buy or use a domain; create an **A** record to the VM’s public IP.
2. Install [Caddy](https://caddyserver.com/docs/install#debian-ubuntu-raspbian) on the VM.
3. Adapt `Caddyfile.example` (replace hostname, keep `reverse_proxy 127.0.0.1:8000`).
4. Tighten the security list: allow **443** from the internet, remove public **8000** if Caddy terminates TLS on 443.

## 10. Updates after `git pull`

```bash
cd ~/YOUR_REPO/infra/oracle
git pull
docker compose up -d --build
```

## Troubleshooting

| Symptom | What to check |
|--------|----------------|
| Cannot SSH | Security list **22** from your IP; correct username/key. |
| Connection timeout on :8000 | Ingress rule for **8000** on subnet; instance has public IP. |
| `docker compose` build OOM | Increase instance RAM or lower parallel jobs; A1 Flex up to free tier max. |
| Out of capacity for A1 | Another region or wait; Oracle free Ampere is capacity-limited. |
| ARM build failure | Paste the failing `pip` package name; may need a pin or x86 VM. |

## Cost

Always Free resources are **free within published limits**; a paid method may still be required at signup. Watch [Oracle Free Tier documentation](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic.htm) for current rules.
