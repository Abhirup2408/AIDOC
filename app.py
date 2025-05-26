import streamlit as st
import google.generativeai as genai
import os
import tempfile

# --- CONFIGURATION ---

# Set your Gemini API Key (preferably as an environment variable)
GEMINI_API_KEY = st.secrets["GOOGLE_API_KEY"]
if not GEMINI_API_KEY:
    st.error("Please set your GEMINI_API_KEY environment variable.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-2.5-flash"  # Or "gemini-1.5-pro" if you have access

# --- UTILITY FUNCTIONS ---

def is_medical_query(query):
    # Basic medical intent filter (expand as needed)
    medical_keywords = [
        "symptom", "disease", "treatment", "medicine", "diagnosis", "doctor",
        "health", "illness", "pain", "fever", "infection", "injury", "test", "scan",
        "medical", "surgery", "prescription", "allergy", "hospital", "clinic",
        "blood", "pressure", "diabetes", "cancer", "asthma", "heart", "lungs", "mental health"
    ]
    query_lower = query.lower()
    return any(word in query_lower for word in medical_keywords)

def get_gemini_response(messages, model=MODEL_NAME, vision=False, file_data=None, file_type=None):
    if vision and file_data:
        # For vision, send image/pdf as part of the input
        parts = [{"text": messages[-1]["content"]}]
        if file_type in ["jpg", "jpeg", "png"]:
            parts.append({"mime_type": f"image/{file_type}", "data": file_data})
        elif file_type == "pdf":
            parts.append({"mime_type": "application/pdf", "data": file_data})
        else:
            parts.append({"mime_type": "application/octet-stream", "data": file_data})
        response = genai.GenerativeModel(model_name=model).generate_content(parts)
        return response.text
    else:
        # For text-only chat
        chat_history = []
        for msg in messages:
            chat_history.append({"role": msg["role"], "parts": [msg["content"]]})
        chat = genai.GenerativeModel(model_name=model).start_chat(history=chat_history)
        user_msg = messages[-1]["content"]
        response = chat.send_message(user_msg)
        return response.text

# --- STREAMLIT UI ---

st.set_page_config(page_title="AI Medical Assistant", page_icon="🩺", layout="centered")

st.title("🩺 AI Medical Assistant")
st.markdown("""
Welcome! This assistant is designed **strictly for medical-related queries**.  
Select a mode below:
""")

# --- MODE SELECTION ---

mode = st.selectbox(
    "Choose a mode:",
    ["Select...", "Student Help", "Doctor Analysis", "Report Result"]
)

st.divider()

# --- STUDENT HELP MODE ---

if mode == "Student Help":
    st.header("🧑‍🎓 Student Help")
    st.info("Ask any **medical-related** question. Non-medical queries will not be answered.")

    if "student_history" not in st.session_state:
        st.session_state.student_history = []

    query = st.text_input("Enter your medical question:", key="student_query")
    if query:
        if not is_medical_query(query):
            st.warning("Please ask only medical-related questions. Non-medical queries are not supported.")
        else:
            st.session_state.student_history.append({"role": "user", "content": query})
            # System prompt to ensure medical context
            system_prompt = (
                "You are a helpful medical assistant. Only answer medical, health, or doctor-related questions. "
                "If the query is not medical, politely refuse to answer."
            )
            messages = [{"role": "user", "content": system_prompt}] + st.session_state.student_history
            with st.spinner("Thinking..."):
                response = get_gemini_response(messages)
            st.session_state.student_history.append({"role": "model", "content": response})
            st.markdown(f"**AI:** {response}")

# --- DOCTOR ANALYSIS MODE ---

elif mode == "Doctor Analysis":
    st.header("👨‍⚕️ Doctor Analysis")
    st.info("Simulated doctor: Step-by-step medical history and possible diagnosis. **Strictly medical only.**")

    if "doctor_history" not in st.session_state:
        st.session_state.doctor_history = []
    if "doctor_phase" not in st.session_state:
        st.session_state.doctor_phase = "start"

    # System prompt for doctor behavior
    doctor_system_prompt = (
        "You are 'MediGuide', a virtual health assistant. "
        "First, ask detailed, step-by-step questions to collect the user's medical history, symptoms, and relevant background. "
        "Ask only one or two questions at a time. After you believe you have sufficient information, ask the user if they wish to proceed to analysis. "
        "If yes, summarize the collected history and discuss possible general conditions or tests that might be relevant, "
        "always reminding the user to consult a real healthcare professional and that this is not a diagnosis. "
        "If the user asks a non-medical question, politely refuse."
    )

    if len(st.session_state.doctor_history) == 0:
        st.session_state.doctor_history.append(
            {"role": "user", "content": doctor_system_prompt}
        )
        st.session_state.doctor_history.append(
            {"role": "model", "content": "Hello, I'm MediGuide, your virtual health assistant. What health concern would you like to discuss today?"}
        )

    for msg in st.session_state.doctor_history[1:]:
        if msg["role"] == "model":
            st.markdown(f"**Doctor:** {msg['content']}")
        else:
            st.markdown(f"**You:** {msg['content']}")

    user_input = st.text_input("Your response:", key="doctor_query")
    if user_input:
        if not is_medical_query(user_input):
            response = "I'm sorry, but I can only assist with medical or health-related queries."
            st.session_state.doctor_history.append({"role": "user", "content": user_input})
            st.session_state.doctor_history.append({"role": "model", "content": response})
            st.warning(response)
        else:
            st.session_state.doctor_history.append({"role": "user", "content": user_input})
            with st.spinner("Doctor is thinking..."):
                response = get_gemini_response(st.session_state.doctor_history)
            st.session_state.doctor_history.append({"role": "model", "content": response})
            st.markdown(f"**Doctor:** {response}")

    st.caption("Disclaimer: This is a simulated assistant for informational purposes only. Always consult a licensed healthcare provider.")

# --- REPORT RESULT MODE ---

elif mode == "Report Result":
    st.header("📄 Report Result")
    st.info("Upload a **medical report** (PDF, JPG, PNG). The AI will attempt to analyze and summarize it. Non-medical documents will not be processed.")

    uploaded_file = st.file_uploader("Upload your medical report (PDF, JPG, PNG):", type=["pdf", "jpg", "jpeg", "png"])

    if uploaded_file:
        file_type = uploaded_file.type.split("/")[-1]
        file_bytes = uploaded_file.read()
        # Basic check for medical content (could be improved with OCR or LLM vision)
        prompt = (
            "This is a medical report. Please analyze and summarize the key medical findings, "
            "tests, and any notable results. If this is not a medical report, respond: "
            "'Sorry, I can only analyze medical reports.'"
        )
        with st.spinner("Analyzing report..."):
            try:
                response = get_gemini_response(
                    [{"role": "user", "content": prompt}],
                    vision=True,
                    file_data=file_bytes,
                    file_type=file_type
                )
                st.markdown(f"**Analysis:** {response}")
            except Exception as e:
                st.error(f"Failed to analyze the report: {e}")

    st.caption("Disclaimer: This is an AI-generated summary for informational purposes only. For medical decisions, consult a licensed healthcare provider.")

# --- DEFAULT / LANDING ---

else:
    st.markdown("""
    ### Welcome to the AI Medical Assistant!
    - **Student Help:** Ask medical questions and get clear answers.
    - **Doctor Analysis:** Simulated doctor will ask you step-by-step history and provide possible insights.
    - **Report Result:** Upload a medical report for AI analysis.
    ---
    **Note:** This tool is for informational and educational purposes only.  
    It does **not** provide medical advice, diagnosis, or treatment.  
    Always consult a qualified healthcare professional for your health concerns.
    """)

