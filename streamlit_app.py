import os, vertexai, streamlit as st
import vertexai.generative_models as genai
from google.cloud import modelarmor_v1

# Google Cloud & Gemini settings
gcp_project_id = os.getenv("PROJECT_ID")
gcp_location = os.getenv("LOCATION", "us-central1")  # Default to 'us-central1' if not set
model_options = {
    "Gemini 1.5 Flash": "gemini-1.5-flash",
    "Gemini 2.0 Flash": "gemini-2.0-flash"
}

# Create the following Model Armor templates beforehand
template_options = {
    "None": "none",
    "All - high only": "ma-all-high",
    "All - medium and above": "ma-all-med",
    "All - low and above": "ma-all-low",
    "Prompt injection and jailbreak - high only": "ma-pijb-high",
    "Prompt injection and jailbreak - medium and above": "ma-pijb-med",
    "Prompt injection and jailbreak - low and above": "ma-pijb-low",
    "Sensitive data protection - basic only": "ma-sdp-basic",
    "Malicious URL detection - only": "ma-mal-url",
    "Responsible AI - high only": "ma-rai-high",
    "Responsible AI - medium and above": "ma-rai-med",
    "Responsible AI - low and above": "ma-rai-low"
}

# Streamlit app config
st.set_page_config(page_title="Model Armor Demo", page_icon="🛡️", initial_sidebar_state="auto")

# Model Armor settings
with st.sidebar:
    st.title("🛡️ Model Armor Demo")
    creds_file = st.file_uploader("Google Cloud credentials file", type="json")
    model_option = st.selectbox("Model", list(model_options.keys()))
    model = model_options[model_option]
    sanitize_request = st.selectbox("Sanitize prompt request?", list(template_options.keys()))
    sanitize_response = st.selectbox("Sanitize model response?", list(template_options.keys()))

# Cache credentials to avoid unnecessary writes
if creds_file is not None and "google_creds" not in st.session_state:
    creds_contents = creds_file.read().decode("utf-8")
    st.session_state.google_creds = creds_contents  # Store creds in session state
    with open("temp_credentials.json", "w") as f:
        f.write(creds_contents)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "temp_credentials.json"

# Initialise Vertex AI and Model Armor only if credentials exist
if creds_file is not None:
    if "vertex_client" not in st.session_state:
        vertexai.init(project=gcp_project_id, location=gcp_location)
        st.session_state.vertex_client = genai.GenerativeModel(model)
    
    if "model_armor_client" not in st.session_state:
        st.session_state.model_armor_client = modelarmor_v1.ModelArmorClient(
            transport="rest",
            client_options={"api_endpoint": "modelarmor.us-central1.rep.googleapis.com"},
        )

# Initialise session state for model and messages
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User-Assistant chat interaction
if creds_file is not None and st.session_state.vertex_client:
    if prompt := st.chat_input("Ask anything"):
        with st.chat_message("user"):
            st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
        if sanitize_request != "None":
            with st.spinner("Analysing prompt request..."):
                try:
                    template_id = template_options[sanitize_request]
                    prompt_data = modelarmor_v1.DataItem()
                    prompt_data.text = prompt
                    request = modelarmor_v1.SanitizeUserPromptRequest(
                        name=f"projects/{gcp_project_id}/locations/{gcp_location}/templates/{template_id}",
                        user_prompt_data=prompt_data,
                    )
                    response = st.session_state.model_armor_client.sanitize_user_prompt(request=request)
                    with st.container(height=300, border=False):
                        with st.expander("Sanitised prompt request", expanded=True):
                            st.write(response)
                except Exception as e:
                    st.error(f"Model Armor error: {e}")

        # Assistant response
        with st.chat_message("assistant"):
            try:
                model_response = st.session_state.vertex_client.generate_content(prompt)
                st.write(model_response.text)
                st.session_state.messages.append({"role": "assistant", "content": model_response.text})
            except Exception as e:
                st.error(f"Vertex AI error: {e}")

        if sanitize_response != "None":
            with st.spinner("Analysing model response..."):
                try:
                    template_id = template_options[sanitize_response]
                    model_data = modelarmor_v1.DataItem()
                    model_data.text = model_response.text
                    request = modelarmor_v1.SanitizeModelResponseRequest(
                        name=f"projects/{gcp_project_id}/locations/{gcp_location}/templates/{template_id}",
                        model_response_data=model_data,
                    )
                    response = st.session_state.model_armor_client.sanitize_model_response(request=request)
                    with st.container(height=300, border=False):
                        with st.expander("Sanitised model response", expanded=True):
                            st.write(response)
                except Exception as e:
                    st.error(f"Model Armor error: {e}")
