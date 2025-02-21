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
    model_option = st.selectbox("Model", list(model_options.keys()))
    model = model_options[model_option]
    sanitize_request = st.selectbox("Sanitize prompt request?", list(template_options.keys()))
    sanitize_response = st.selectbox("Sanitize model response?", list(template_options.keys()))

# Initialise Vertex AI and Model Armor
if "vertex_client" not in st.session_state:
    vertexai.init(project=gcp_project_id, location=gcp_location)
    st.session_state.vertex_client = genai.GenerativeModel(model)

if "model_armor_client" not in st.session_state:
    st.session_state.model_armor_client = modelarmor_v1.ModelArmorClient(
        transport="rest",
        client_options={"api_endpoint": "modelarmor.us-central1.rep.googleapis.com"},
    )

# Initialise session state for chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get readable match state message
def get_match_state_message(match_state):
    return "No Match Found ✅" if match_state == 1 else "Match Found 🚨" if match_state == 2 else "Unknown"

# User-Assistant chat interaction
if st.session_state.vertex_client and st.session_state.model_armor_client:
    if prompt := st.chat_input("Ask anything"):
        with st.chat_message("user"):
            st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
        if sanitize_request != "None":
            try:
                template_id = template_options[sanitize_request]
                prompt_data = modelarmor_v1.DataItem()
                prompt_data.text = prompt
                request = modelarmor_v1.SanitizeUserPromptRequest(
                    name=f"projects/{gcp_project_id}/locations/{gcp_location}/templates/{template_id}",
                    user_prompt_data=prompt_data,
                )
                response = st.session_state.model_armor_client.sanitize_user_prompt(request=request)
                
                if response.sanitization_result.filter_match_state == 2:
                    with st.container(border=True):
                        sdp_match_state = response.sanitization_result.filter_results["sdp"].sdp_filter_result.inspect_result.match_state
                        pi_and_jailbreak_match_state = response.sanitization_result.filter_results["pi_and_jailbreak"].pi_and_jailbreak_filter_result.match_state
                        malicious_uris_match_state = response.sanitization_result.filter_results["malicious_uris"].malicious_uri_filter_result.match_state
                        rai_match_state = response.sanitization_result.filter_results["rai"].rai_filter_result.match_state
                        rai_sexually_explicit_match_state = response.sanitization_result.filter_results["rai"].rai_filter_result.rai_filter_type_results["sexually_explicit"].match_state
                        rai_hate_speech_match_state = response.sanitization_result.filter_results["rai"].rai_filter_result.rai_filter_type_results["hate_speech"].match_state
                        rai_harassment_match_state = response.sanitization_result.filter_results["rai"].rai_filter_result.rai_filter_type_results["harassment"].match_state
                        rai_dangerous_match_state = response.sanitization_result.filter_results["rai"].rai_filter_result.rai_filter_type_results["dangerous"].match_state

                        st.write(f"**Sensitive Data Protection**: {get_match_state_message(sdp_match_state)}")
                        st.write(f"**Prompt Injection and Jailbreak**: {get_match_state_message(pi_and_jailbreak_match_state)}")
                        st.write(f"**Malicious URIs**: {get_match_state_message(malicious_uris_match_state)}")
                        st.write(f"**Responsible AI**: {get_match_state_message(rai_match_state)}")
                        st.write(f"* **Sexually Explicit**: {get_match_state_message(rai_sexually_explicit_match_state)}")
                        st.write(f"* **Hate Speech**: {get_match_state_message(rai_hate_speech_match_state)}")
                        st.write(f"* **Harassment**: {get_match_state_message(rai_harassment_match_state)}")
                        st.write(f"* **Dangerous**: {get_match_state_message(rai_dangerous_match_state)}")
                    with st.expander("Sanitised prompt request (raw)", expanded=False):
                        with st.container(height=300, border=True):
                            st.write(response)
                    st.stop()
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
                    
                    if response.sanitization_result.filter_match_state == 2:
                        with st.container(border=True):
                            sdp_match_state = response.sanitization_result.filter_results["sdp"].sdp_filter_result.inspect_result.match_state
                            pi_and_jailbreak_match_state = response.sanitization_result.filter_results["pi_and_jailbreak"].pi_and_jailbreak_filter_result.match_state
                            malicious_uris_match_state = response.sanitization_result.filter_results["malicious_uris"].malicious_uri_filter_result.match_state
                            rai_match_state = response.sanitization_result.filter_results["rai"].rai_filter_result.match_state
                            rai_sexually_explicit_match_state = response.sanitization_result.filter_results["rai"].rai_filter_result.rai_filter_type_results["sexually_explicit"].match_state
                            rai_hate_speech_match_state = response.sanitization_result.filter_results["rai"].rai_filter_result.rai_filter_type_results["hate_speech"].match_state
                            rai_harassment_match_state = response.sanitization_result.filter_results["rai"].rai_filter_result.rai_filter_type_results["harassment"].match_state
                            rai_dangerous_match_state = response.sanitization_result.filter_results["rai"].rai_filter_result.rai_filter_type_results["dangerous"].match_state

                            st.write(f"**Sensitive Data Protection**: {get_match_state_message(sdp_match_state)}")
                            st.write(f"**Prompt Injection and Jailbreak**: {get_match_state_message(pi_and_jailbreak_match_state)}")
                            st.write(f"**Malicious URIs**: {get_match_state_message(malicious_uris_match_state)}")
                            st.write(f"**Responsible AI**: {get_match_state_message(rai_match_state)}")
                            st.write(f"* **Sexually Explicit**: {get_match_state_message(rai_sexually_explicit_match_state)}")
                            st.write(f"* **Hate Speech**: {get_match_state_message(rai_hate_speech_match_state)}")
                            st.write(f"* **Harassment**: {get_match_state_message(rai_harassment_match_state)}")
                            st.write(f"* **Dangerous**: {get_match_state_message(rai_dangerous_match_state)}")
                        with st.expander("Sanitised prompt request (raw)", expanded=False):
                            with st.container(height=300, border=True):
                                st.write(response)
                        st.stop()
                except Exception as e:
                    st.error(f"Model Armor error: {e}")
