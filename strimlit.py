import streamlit as st
import requests

st.title("🎵 Audio Upload with LLM API Call")

# File uploader
audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg", "m4a"])

# Submit button
if st.button("Submit"):
    if audio_file is not None:
        st.success(f"Uploaded file: {audio_file.name}")
        st.audio(audio_file, format="audio/wav")

        # Example: sending file to an LLM API
        with st.spinner("Sending to LLM API..."):
            # Replace this with your actual LLM endpoint
            api_url = "https://your-llm-endpoint.com/analyze"

            files = {"file": (audio_file.name, audio_file, audio_file.type)}
            response = requests.post(api_url, files=files)

            if response.status_code == 200:
                st.subheader("LLM Response:")
                st.write(response.json())  # or response.text
            else:
                st.error(f"API call failed: {response.status_code}")
    else:
        st.warning("Please upload a file before submitting.")
