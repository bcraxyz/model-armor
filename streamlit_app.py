import os, vertexai, streamlit as st
import vertexai.generative_models as genai
from google.cloud import modelarmor_v1
from anthropic import AnthropicVertex
from openai import OpenAI

# Ensure the following Model Armor templates are available in the specified Google Cloud project
# "None": "none"
# "All - high only": "ma-all-high"
# "All - medium and above": "ma-all-med"
# "All - low and above": "ma-all-low"
# "Prompt injection and jailbreak - high only": "ma-pijb-high"
# "Prompt injection and jailbreak - medium and above": "ma-pijb-med"
# "Prompt injection and jailbreak - low and above": "ma-pijb-low"
# "Sensitive data protection - basic only": "ma-sdp-basic"
# "Malicious URL detection - only": "ma-mal-url"
# "Responsible AI - high only": "ma-rai-high"
# "Responsible AI - medium and above": "ma-rai-med"
# "Responsible AI - low and above": "ma-rai-low"

model_options = [
    {"name": "gemini-2.5-flash", "display_name": "Gemini 2.5 Flash", "provider": "Google", "location": "global"},
    {"name": "gemini-2.5-flash-lite", "display_name": "Gemini 2.5 Flash Lite", "provider": "Google", "location": "global"},
    {"name": "claude-sonnet-4@20250514", "display_name": "Claude Sonnet 4", "provider": "Anthropic", "location": "us-east5"},
    {"name": "gpt-4o-mini", "display_name": "GPT-4o mini", "provider": "OpenAI", "location": "global"},
]

model_armor_endpoints = [
    {"location": "us-central1", "endpoint": "modelarmor.us-central1.rep.googleapis.com"},
    {"location": "asia-southeast1", "endpoint": "modelarmor.asia-southeast1.rep.googleapis.com"},
]

# Initialise session state for key variables
if "openai_api_key" not in st.session_state :
    st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")
if "project_id" not in st.session_state:
    st.session_state.project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID", "")
if "location" not in st.session_state:
    st.session_state.location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
if "google_creds" not in st.session_state:
    st.session_state.google_creds = None
if "model" not in st.session_state:
    st.session_state.model = None
if "endpoint" not in st.session_state:
    st.session_state.endpoint = None
if "messages" not in st.session_state:
    st.session_state.messages = []

def reset_clients():
    st.session_state.pop("vertex_client", None)
    st.session_state.pop("anthropic_client", None)
    st.session_state.pop("model_armor_client", None)
    st.session_state.model = None

# Streamlit app config
st.set_page_config(page_title="Model Armor Demo", page_icon="🛡️", initial_sidebar_state="auto")

# Model Armor settings
with st.sidebar:
    st.title("🛡️ Model Armor Demo")
    with st.expander("**⚙️ Model Settings**", expanded=False):
        selected_model = st.selectbox("**Model**", options=model_options, format_func=lambda m: m["display_name"])
        model = selected_model["name"]
        provider = selected_model["provider"]

        creds_file = st.file_uploader("**Google Cloud credentials file**", type="json") 

        if provider == "OpenAI":
            st.text_input("OpenAI API key", type="password", key="openai_api_key")
        
    with st.expander("**⚙️ Model Armor Settings**", expanded=True):
        with st.expander("**⚙️ Project Settings**", expanded=True):
            project_id = st.text_input("**Project ID**")
            selected_location = st.selectbox("**Location**", options=model_armor_endpoints, format_func=lambda m: m["location"])
            location = selected_location["location"]
            endpoint = selected_location["endpoint"]
        
        with st.expander("**⚙️ Detection Settings**", expanded=True):
            detection_type = None
            confidence_level = None
            sanitize_request = st.checkbox("Sanitize prompt request?")
            
            if sanitize_request:
                detection_type = st.radio(
                    "**Detection type**",
                    [
                        "Malicious URLs",
                        "Sensitive data protection",
                        "Prompt injection and jailbreak",
                        "Responsible AI",
                        "All of the above"
                    ]
                )

                if detection_type in [
                    "Prompt injection and jailbreak",
                    "Responsible AI",
                    "All of the above"
                ]:
                    confidence_level = st.radio(
                        "**Confidence level**",
                        ["High only", "Medium and above", "Low and above"]
                    )

            # Map detection_type and confidence_level to templates
            if detection_type == "Malicious URLs":
                template_id = "ma-mal-url"
            elif detection_type == "Sensitive data protection":
                template_id = "ma-sdp-basic"
            elif detection_type == "Prompt injection and jailbreak":
                if confidence_level == "High only":
                    template_id = "ma-pijb-high"
                elif confidence_level == "Medium and above":
                    template_id = "ma-pijb-med"
                elif confidence_level == "Low and above":
                    template_id = "ma-pijb-low"
            elif detection_type == "Responsible AI":
                if confidence_level == "High only":
                    template_id = "ma-rai-high"
                elif confidence_level == "Medium and above":
                    template_id = "ma-rai-med"
                elif confidence_level == "Low and above":
                    template_id = "ma-rai-low"
            elif detection_type == "All of the above":
                if confidence_level == "High only":
                    template_id = "ma-all-high"
                elif confidence_level == "Medium and above":
                    template_id = "ma-all-med"
                elif confidence_level == "Low and above":
                    template_id = "ma-all-low"

            sanitize_response = st.checkbox("Sanitize model response?", help="Uses `All - low and above` template")

# Cache credentials to avoid unnecessary writes
if creds_file is not None:
    creds_contents = creds_file.read().decode("utf-8")
    if creds_contents != st.session_state.get("google_creds"):
        st.session_state.google_creds = creds_contents  # Store creds in session state
        with open("temp_credentials.json", "w") as f:
            f.write(creds_contents)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "temp_credentials.json"
        reset_clients()

# Check if Project ID, Location, Model or Endpoint has changed
if project_id != st.session_state.project_id or location != st.session_state.location:
    reset_clients()
    st.session_state.project_id = project_id
    st.session_state.location = location

if model != st.session_state.model:
    reset_clients()
    st.session_state.model = model

if endpoint != st.session_state.endpoint:
    reset_clients()
    st.session_state.endpoint = endpoint

# Print readable match state message
def get_match_state_message(match_state):
    return "No Match Found ✅" if match_state == 1 else "Match Found 🚨" if match_state == 2 else "*Not Assessed*"

def print_results(response):
    if "sdp" in response.sanitization_result.filter_results:
        sdp_match_state = response.sanitization_result.filter_results["sdp"].sdp_filter_result.inspect_result.match_state
    else:
        sdp_match_state = None
    if "pi_and_jailbreak" in response.sanitization_result.filter_results:
        pi_and_jailbreak_match_state = response.sanitization_result.filter_results["pi_and_jailbreak"].pi_and_jailbreak_filter_result.match_state
    else:
        pi_and_jailbreak_match_state = None
    if "malicious_uris" in response.sanitization_result.filter_results:
        malicious_uris_match_state = response.sanitization_result.filter_results["malicious_uris"].malicious_uri_filter_result.match_state
    else:
        malicious_uris_match_state = None
    if "rai" in response.sanitization_result.filter_results:
        rai_match_state = response.sanitization_result.filter_results["rai"].rai_filter_result.match_state
        rai_sexually_explicit_match_state = response.sanitization_result.filter_results["rai"].rai_filter_result.rai_filter_type_results["sexually_explicit"].match_state
        rai_hate_speech_match_state = response.sanitization_result.filter_results["rai"].rai_filter_result.rai_filter_type_results["hate_speech"].match_state
        rai_harassment_match_state = response.sanitization_result.filter_results["rai"].rai_filter_result.rai_filter_type_results["harassment"].match_state
        rai_dangerous_match_state = response.sanitization_result.filter_results["rai"].rai_filter_result.rai_filter_type_results["dangerous"].match_state
    else:
        rai_match_state = None
        rai_sexually_explicit_match_state = None
        rai_hate_speech_match_state = None
        rai_harassment_match_state = None
        rai_dangerous_match_state = None

    st.write(f"**Sensitive Data Protection**: {get_match_state_message(sdp_match_state)}")
    st.write(f"**Prompt Injection and Jailbreak**: {get_match_state_message(pi_and_jailbreak_match_state)}")
    st.write(f"**Malicious URIs**: {get_match_state_message(malicious_uris_match_state)}")
    st.write(f"**Responsible AI**: {get_match_state_message(rai_match_state)}")
    st.write(f"* **Sexually Explicit**: {get_match_state_message(rai_sexually_explicit_match_state)}")
    st.write(f"* **Hate Speech**: {get_match_state_message(rai_hate_speech_match_state)}")
    st.write(f"* **Harassment**: {get_match_state_message(rai_harassment_match_state)}")
    st.write(f"* **Dangerous**: {get_match_state_message(rai_dangerous_match_state)}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User-Assistant chat interaction
if prompt := st.chat_input("Ask anything"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    if (provider in ["Google", "Anthropic"]):
        if not st.session_state.google_creds:
            st.error("Please upload the Google Cloud credentials file.")
            reset_clients()
            st.stop()
        elif not st.session_state.project_id.strip() or not st.session_state.location.strip():
            st.error("Please provide Google Cloud Project ID.")
            reset_clients()
            st.stop()
    elif (provider == "OpenAI" and not st.session_state.openai_api_key):
        st.error("Please provide the OpenAI API key.")
        st.session_state.model = None
        st.stop()
    
    if provider == "Google" and ("vertex_client" not in st.session_state or st.session_state.get("model") != model):
        try:
            vertexai.init(project=st.session_state.project_id, location=st.session_state.location)
            st.session_state.vertex_client = genai.GenerativeModel(model)
            st.session_state.model = model
        except Exception as e:
            st.error(f"Failed to initialize Vertex AI client: {e}")
            st.stop()
    
    if provider == "Anthropic" and ("anthropic_client" not in st.session_state or st.session_state.get("model") != model):
        try:
            st.session_state.anthropic_client = AnthropicVertex(project_id=st.session_state.project_id, region=selected_model.get("location"))
            st.session_state.model = model
        except Exception as e:
            st.error(f"Failed to initialize Anthropic client: {e}")
            st.stop()

    if provider == "OpenAI" and ("openai_client" not in st.session_state or st.session_state.get("model") != model):
        try:
            st.session_state.openai_client = OpenAI(api_key=st.session_state.openai_api_key)
            st.session_state.model = model
        except Exception as e:
            st.error(f"Failed to initialize OpenAI client: {e}")
            st.stop()

    if (sanitize_request or sanitize_response) and "model_armor_client" not in st.session_state:
        try:
            st.session_state.model_armor_client = modelarmor_v1.ModelArmorClient(
                transport="rest",
                client_options={"api_endpoint": endpoint},
            )
        except Exception as e:
            st.error(f"Failed to initialize Model Armor client: {e}")
            st.stop()

    with st.chat_message("user"):
        st.markdown(prompt)
        
    if sanitize_request:
        try:
            with st.spinner("Analysing prompt request..."):    
                prompt_data = modelarmor_v1.DataItem(text=prompt)
                request = modelarmor_v1.SanitizeUserPromptRequest(
                    name=f"projects/{st.session_state.project_id}/locations/{st.session_state.location}/templates/{template_id}",
                    user_prompt_data=prompt_data,
                )
                response = st.session_state.model_armor_client.sanitize_user_prompt(request=request)
                
            if response.sanitization_result.filter_match_state == 2:
                with st.container(border=True):
                    print_results(response)
                with st.expander("Sanitised prompt request (raw)", expanded=False):
                    with st.container(height=300, border=True):
                        st.write(response)
                st.stop()
        
        except Exception as e:
            st.error(f"Model Armor error during request sanitisation: {e}")
            st.stop()

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            try:
                if provider == "Google":
                    response = st.session_state.vertex_client.generate_content(prompt)
                    model_response = response.text
                elif provider == "Anthropic":
                    response = st.session_state.anthropic_client.messages.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1024
                    )
                    model_response = response.content[0].text
                elif provider == "OpenAI":
                    response = st.session_state.openai_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    model_response = response.choices[0].message.content     
            
            except Exception as e:
                st.error(f"Error while generating LLM response: {e}")
                st.stop()
                
        st.markdown(model_response)
        st.session_state.messages.append({"role": "assistant", "content": model_response})

    if sanitize_response:
        try:
            with st.spinner("Analysing model response..."):    
                response_template_id = "ma-all-low"
                model_data = modelarmor_v1.DataItem(text=model_response)
                request = modelarmor_v1.SanitizeModelResponseRequest(
                    name=f"projects/{st.session_state.project_id}/locations/{st.session_state.location}/templates/{response_template_id}",
                    model_response_data=model_data,
                )
                response = st.session_state.model_armor_client.sanitize_model_response(request=request)
                
            if response.sanitization_result.filter_match_state == 2:
                with st.container(border=True):
                    print_results(response)
                with st.expander("Sanitised model response (raw)", expanded=False):
                    with st.container(height=300, border=True):
                        st.write(response)
                st.stop()
        
        except Exception as e:
            st.error(f"Model Armor error during response sanitisation: {e}")
            st.stop()
    
    st.rerun()
