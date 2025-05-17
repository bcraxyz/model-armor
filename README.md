# 🛡️ Model Armor Demo
A Streamlit chatbot for testing Google Cloud Model Armor LLM safety and security offering. 

### Features

- Supports `gemini-1.5-flash` and `gemini-2.0-flash` models via Vertex AI, and `gpt-4o-mini` model via OpenAI
- Supports two modes of deployment:
  - `cloud_run.py`: For deployment on Google Cloud Run, will use Application Default Credentials
  - `streamlit_app.py`: For off-Google Cloud deployment, requires Google Cloud service account credentials
- Enable **prompt sanitization**, with optional **response sanitization**, for the following detection types
  - Malicious URLs
  - Sensitive data protection
  - Prompt injection & jailbreak
  - Responsible AI
  - All of the above
- Configure **confidence levels** (high only / medium & above / low & above)
- Display detailed sanitization results inline (e.g., hate speech, explicit content)

![model-armor-demo](./model-armor-demo.png)

### Setup

1. Clone the repo & install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Environment variables required:

    - `GOOGLE_CLOUD_PROJECT_ID`: Google Cloud project ID
    - `GOOGLE_CLOUD_LOCATION`: Google Cloud location (default: `us-central1`)
    - `MODEL_ARMOR_ENDPOINT`: Model Armor endpoint (default: `modelarmor.us-central1.rep.googleapis.com`)
    - `OPENAI_API_KEY` (optional): OpenAI API key (if you intend to use both model providers)

3. Prepare Model Armor templates in your Google Cloud project:

    Ensure the following templates are created and published in Model Armor:

    - "All - high only": `ma-all-high`
    - "All - medium and above": `ma-all-med`
    - "All - low and above": `ma-all-low`
    - "Prompt injection and jailbreak - high only": `ma-pijb-high`
    - "Prompt injection and jailbreak - medium and above": `ma-pijb-med`
    - "Prompt injection and jailbreak - low and above": `ma-pijb-low`
    - "Sensitive data protection - basic only": `ma-sdp-basic`
    - "Malicious URL detection - only": `ma-mal-url`
    - "Responsible AI - high only": `ma-rai-high`
    - "Responsible AI - medium and above": `ma-rai-med`
    - "Responsible AI - low and above": `ma-rai-low`

4. Run the app:

    ```bash
    streamlit run streamlit_app.py
    ```
