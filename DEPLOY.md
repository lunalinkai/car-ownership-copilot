# Deploying to Streamlit Community Cloud

This app is ready to deploy as-is. The repo already contains everything Streamlit
Community Cloud needs: `app.py` (entry point), `requirements.txt`, and
`.streamlit/config.toml` (theme).

## Steps (~5 clicks, no CLI)

1. Go to **https://share.streamlit.io** and sign in with the **GitHub account that owns this repo** (`lunalinkai`).
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `lunalinkai/car-ownership-copilot`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. *(Optional but recommended)* Open **Advanced settings** → set **Python version** to **3.12**.
5. Click **Deploy**. First build takes ~2–3 minutes while it installs dependencies.

You'll get a public URL like `https://<your-subdomain>.streamlit.app` — paste that
into the Jerry application.

## API key

This deploy uses the **visitor-paste** model: there's a key field in the sidebar,
and each visitor supplies their own OpenAI key (used only for their session, never
stored). So **you do not need to add any secrets.**

If you'd rather have the demo "just work" without visitors needing a key, add your
key under **App settings → Secrets** in the Streamlit Cloud dashboard:

```toml
OPENAI_API_KEY = "sk-..."
```

⚠️ If you do that, every visitor's usage is billed to your OpenAI account — set a
hard spend limit at https://platform.openai.com/account/limits first.

## Updating the deployed app

Streamlit Cloud auto-redeploys on every push to `main`. Just:

```bash
git add -A && git commit -m "..." && git push
```
