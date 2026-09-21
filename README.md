# 🧑‍💼 Job Finding AI Agent

A beginner-friendly, single-agent **CrewAI** application that reads your
LinkedIn profile PDF, asks about your job preferences (remote or onsite,
and where), then searches the live web to find real, matching job openings.

- **Agent framework:** [CrewAI](https://docs.crewai.com/) (single agent)
- **LLM:** Groq's free `openai/gpt-oss-120b` model
- **Search:** DuckDuckGo via the `ddgs` package — **100% free, no API key**
- **Web reading:** CrewAI's built-in `ScrapeWebsiteTool` — free, no API key
- **UI:** Streamlit
- **PDF parsing:** `pypdf`

## How it works

1. You upload your LinkedIn "Save to PDF" profile export.
2. The app extracts the text from it.
3. You tell it whether you want a **Remote** or **Onsite** job (and, if
   onsite, your own location, or "Anywhere in England" / "Anywhere in the
   United States").
4. A single CrewAI agent (the "Job Search Specialist") uses free web search
   and web-page scraping tools to look across job boards, company career
   pages, and articles, then returns a curated Markdown list of 8-12 real
   matching jobs with links.

## Project structure

```
job-finding-ai-agent/
├── app.py                      # Streamlit UI (entry point)
├── agent.py                    # CrewAI agent, task, and crew definition
├── tools/
│   ├── __init__.py
│   └── duckduckgo_tool.py      # Free DuckDuckGo search tool (no API key)
├── utils/
│   ├── __init__.py
│   └── pdf_parser.py           # Extracts text from the uploaded PDF
├── .streamlit/
│   └── secrets.toml.example    # Template for Streamlit Cloud secrets
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## 1. Get a free Groq API key

1. Go to <https://console.groq.com/keys>
2. Sign up (it's free) and create an API key.
3. Keep it handy — you'll paste it into the app's sidebar (or into a
   `.env` file / Streamlit secrets, see below).

## 2. Run it locally

```bash
# 1. Clone your repo
git clone https://github.com/<your-username>/job-finding-ai-agent.git
cd job-finding-ai-agent

# 2. Create a virtual environment (Python 3.10–3.13)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) set your Groq key so you don't have to paste it every time
cp .env.example .env
# then edit .env and add your real GROQ_API_KEY

# 5. Run the app
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`), upload
your LinkedIn PDF, set your preferences, and click **Find Matching Jobs**.

If you didn't set up `.env`, just paste your Groq API key into the
**Groq API Key** field in the sidebar — it's only kept for your session.

## 3. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: Job Finding AI Agent"
git branch -M main
git remote add origin https://github.com/<your-username>/job-finding-ai-agent.git
git push -u origin main
```

`.env` and `.streamlit/secrets.toml` are already listed in `.gitignore`, so
your real API key will **never** be pushed to GitHub.

## 4. Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. Click **New app**, pick your `job-finding-ai-agent` repo, branch `main`,
   and set the main file to `app.py`.
3. Before (or right after) deploying, open **⋮ → Settings → Secrets** for
   the app and paste:
   ```toml
   GROQ_API_KEY = "your_real_groq_api_key"
   ```
4. Click **Deploy**. Streamlit Cloud will install everything from
   `requirements.txt` and start the app.

The app will automatically pick up `GROQ_API_KEY` from Streamlit secrets,
so users won't need to paste a key themselves (though the sidebar field
still lets anyone override it with their own key).

## Notes for beginners

- **Why `openai/gpt-oss-120b` via a `base_url`?** Groq's API is
  OpenAI-compatible, and routing through CrewAI's `openai/` provider path
  with a custom `base_url` is currently the most reliable way to call Groq
  models from CrewAI (it avoids a known bug in CrewAI's LiteLLM-based
  `groq/` routing). You don't need to change anything — it's already wired
  up in `agent.py`.
- **Why DuckDuckGo instead of Google/Serper?** Serper-based search tools
  need a paid/rate-limited API key. `ddgs` (DuckDuckGo Search) is free and
  needs no key at all, which keeps this project beginner-friendly and free
  to run.
- **First run is slow-ish**: the agent does several searches and reads a
  few web pages, so expect roughly 30–90 seconds per search.
- **Rate limits**: both Groq's free tier and DuckDuckGo have generous but
  not unlimited free usage — if you hit an error, wait a bit and try again.
