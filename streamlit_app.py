import os, vertexai, streamlit as st
import vertexai.generative_models as genai
from google.cloud import modelarmor_v1
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

# Google Cloud & Gemini settings
GOOGLE_CLOUD_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
MODEL_ARMOR_ENDPOINT = os.getenv("MODEL_ARMOR_ENDPOINT", "modelarmor.us-central1.rep.googleapis.com")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not GOOGLE_CLOUD_PROJECT_ID:
    raise EnvironmentError("GOOGLE_CLOUD_PROJECT_ID  must be set when using Vertex AI.")

model_options = {
    "Gemini 1.5 Flash": "gemini-1.5-flash",
    "Gemini 2.0 Flash": "gemini-2.0-flash",
    "GPT-4o mini": "gpt-4o-mini"
}

model_provider = {
    "gemini-1.5-flash": "Vertex AI",
    "gemini-2.0-flash": "Vertex AI",
    "gpt-4o-mini": "OpenAI"
}

# Initialise session state for Open AI key
if "openai_api_key" not in st.session_state :
    st.session_state.openai_api_key = ""

# Streamlit app config
st.set_page_config(page_title="Model Armor Demo", page_icon="🛡️", initial_sidebar_state="auto")

# Model Armor settings
with st.sidebar:
    st.title("🛡️ Model Armor Demo")
    with st.expander("**⚙️ Model Settings**", expanded=False):
        model_option = st.selectbox("Model", list(model_options.keys()))
        model = model_options[model_option]    
        creds_file = st.file_uploader("Google Cloud credentials file", type="json") 

        if model_provider[model] == "OpenAI":
            openai_api_key = st.text_input("OpenAI API key", type="password", value=st.session_state.openai_api_key)
            if openai_api_key != st.session_state.openai_api_key:
                st.session_state.openai_api_key = openai_api_key
                # Force re-initialization by removing the old client
                if "openai_client" in st.session_state:
                    del st.session_state.openai_client
    
    with st.expander("**⚙️ Model Armor Settings**", expanded=True):
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

    with st.expander("**⚙️ Llama Guard Settings**", expanded=False):
        include_llama_guard = st.checkbox("Include Llama Guard?", disabled=True)
        # groq_api_key = st.text_input("Groq API key", type="password", help="Get your API key [here](https://console.groq.com/keys).")

# Cache credentials to avoid unnecessary writes
if creds_file is not None:
    creds_contents = creds_file.read().decode("utf-8")
    if creds_contents != st.session_state.get("google_creds"):
        st.session_state.google_creds = creds_contents  # Store creds in session state
        with open("temp_credentials.json", "w") as f:
            f.write(creds_contents)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "temp_credentials.json"

        # If creds change, clients might need re-init
        if "vertex_client" in st.session_state: 
            del st.session_state.vertex_client
        if "model_armor_client" in st.session_state: 
            del st.session_state.model_armor_client

# Initialise Vertex AI and Model Armor clients if credentials exist in session state
if "google_creds" in st.session_state:
    vertex_model = st.session_state.get("vertex_model")
    if "vertex_client" not in st.session_state or vertex_model != model:
        try:
            vertexai.init(project=GOOGLE_CLOUD_PROJECT_ID, location=GOOGLE_CLOUD_LOCATION)
            st.session_state.vertex_client = genai.GenerativeModel(model)
            st.session_state.vertex_model = model
        except Exception as e:
            st.error(f"Failed to initialize Vertex AI client: {e}")
            st.stop()

    if "model_armor_client" not in st.session_state:
        try:
            st.session_state.model_armor_client = modelarmor_v1.ModelArmorClient(
                transport="rest",
                client_options={"api_endpoint": MODEL_ARMOR_ENDPOINT},
            )
        except Exception as e:
            st.error(f"Failed to initialize Model Armor client: {e}")
            st.stop()

# Initialise OpenAI client if API key available
if model_provider[model] == "OpenAI":
    if st.session_state.openai_api_key:
        current_openai_api_key = getattr(st.session_state.get("openai_client"), 'api_key', None)
        if "openai_client" not in st.session_state or (current_openai_api_key != st.session_state.openai_api_key):
            try:
                st.session_state.openai_client = OpenAI(api_key=st.session_state.openai_api_key)
            except Exception as e:
                st.error(f"Failed to initialize OpenAI client: {e}")
                st.stop()

# Initialise session state for model and messages
if "messages" not in st.session_state:
    st.session_state.messages = []

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
    if not creds_file:
        st.error("Please upload the Google Cloud credentials file")
    else:
        with st.chat_message("user"):
            st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
        if sanitize_request:
            try:
                prompt_data = modelarmor_v1.DataItem(text=prompt)
                request = modelarmor_v1.SanitizeUserPromptRequest(
                    name=f"projects/{GOOGLE_CLOUD_PROJECT_ID}/locations/{GOOGLE_CLOUD_LOCATION}/templates/{template_id}",
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
                st.error(f"Model Armor error: {e}")

        # Assistant response
        with st.chat_message("assistant"):
            if model_provider[model] == "Vertex AI":
                try:
                    response = st.session_state.vertex_client.generate_content(prompt)
                    model_response = response.text
                except Exception as e:
                    st.error(f"Vertex AI error: {e}")
            elif model_provider[model] == "OpenAI":
                if not openai_api_key:
                    st.error("Please provide your OpenAI API key.")
                    st.stop()
                
                try:
                    response = st.session_state.openai_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    model_response = response.choices[0].message.content
                except Exception as e:
                    st.error(f"OpenAI error: {e}")
                    st.stop()

            st.markdown(model_response)
            st.session_state.messages.append({"role": "assistant", "content": model_response})

        if sanitize_response:
            with st.spinner("Analysing model response..."):
                try:
                    template_id = "ma-all-low"
                    model_data = modelarmor_v1.DataItem(text=model_response)
                    request = modelarmor_v1.SanitizeModelResponseRequest(
                        name=f"projects/{GOOGLE_CLOUD_PROJECT_ID}/locations/{GOOGLE_CLOUD_LOCATION}/templates/{template_id}",
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
                    st.error(f"Model Armor error: {e}")
