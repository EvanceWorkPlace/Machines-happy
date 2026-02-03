# Aviator Betting Analysis (minimal scaffold)

This project records Aviator crash results, analyzes history, and returns AI-based cashout suggestions.

Quick start (development):

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run migrations and start the server:

```bash
python manage.py migrate
python manage.py runserver
```

3. Open http://127.0.0.1:8000/ to view the dashboard.

Notes:
- This is a minimal scaffold. The AI helper is a heuristic in `aviator/ai.py` and includes placeholders for LangChain/OpenAI integration.
- To enable OpenAI, add integration code and API keys securely (not included here).
# Machines-happy