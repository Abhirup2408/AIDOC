import streamlit as st
import google.generativeai as genai
import os

# --- CONFIGURATION ---

# Set your Gemini API Key (preferably as an environment variable or Streamlit secrets)
GEMINI_API_KEY = st.secrets["GOOGLE_API_KEY"] 
if not GEMINI_API_KEY:
    st.error("Please set your GEMINI_API_KEY environment variable or Streamlit secret.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-1.5-flash"  # Or "gemini-1.5-pro" if you have access

# --- UTILITY FUNCTIONS ---

def is_medical_query(query):
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
            system_prompt = (
                "You are a helpful medical assistant. Only answer medical, health, or doctor-related questions. "
                "If the query is not medical, politely refuse to answer."
            )
            messages = [{"role": "user", "content": system_prompt}] + st.session_state.student_history
            with st.spinner("Thinking..."):
                response = get_gemini_response(messages)
            st.session_state.student_history.append({"role": "model", "content": response})
            st.markdown(f"**AI:** {response}")

# --- DOCTOR ANALYSIS MODE (STEPWISE, STRUCTURED) ---

elif mode == "Doctor Analysis":
    st.header("👨‍⚕️ Doctor Analysis")
    st.info("Simulated doctor: Step-by-step medical history and possible diagnosis. **Strictly medical only.**")

    # Define the clinical interview steps and their questions
    clinical_steps = [
        ("Chief Complaint", "What brings you in today? What is your main concern?"),
        ("HPI_Onset", "When did this problem start?"),
        ("HPI_Location", "Where is the symptom located?"),
        ("HPI_Duration", "How long does it last? Is it constant or intermittent?"),
        ("HPI_Character", "What does it feel like (e.g., sharp, dull, throbbing, burning)?"),
        ("HPI_Aggravating", "What makes it worse?"),
        ("HPI_Relieving", "What makes it better?"),
        ("HPI_Timing", "Does it occur at a specific time of day?"),
        ("HPI_Severity", "On a scale of 0-10, how bad is it?"),
        ("HPI_Associated", "Are there any other symptoms accompanying the main problem?"),
        ("PMH", "Do you have any chronic conditions, past illnesses, surgeries, or hospitalizations?"),
        ("Medications", "What medications are you currently taking? Any allergies?"),
        ("Family History", "Any significant diseases in your family (e.g., heart disease, diabetes)?"),
        ("Social History", "Do you smoke, drink alcohol, use recreational drugs? What is your occupation and living situation?"),
        ("Review of Systems", "Do you have any other symptoms in other body systems (e.g., fever, cough, rashes, joint pain, etc.)?"),
    ]

    if "doctor_step" not in st.session_state:
        st.session_state.doctor_step = 0
    if "doctor_answers" not in st.session_state:
        st.session_state.doctor_answers = {}

    current_step = st.session_state.doctor_step

    if current_step < len(clinical_steps):
        step_name, step_question = clinical_steps[current_step]
        st.markdown(f"**{step_name.replace('_', ' ')}**")
        user_input = st.text_input(step_question, key=f"step_{current_step}")
        if user_input:
            if not is_medical_query(user_input) and current_step == 0:
                st.warning("Please answer with your main medical concern.")
            else:
                st.session_state.doctor_answers[step_name] = user_input
                st.session_state.doctor_step += 1
                st.experimental_rerun()
        # Show previous Q&A
        for idx in range(current_step):
            prev_name, prev_question = clinical_steps[idx]
            prev_answer = st.session_state.doctor_answers.get(prev_name, "")
            st.markdown(f"**{prev_name.replace('_', ' ')}:** {prev_answer}")
    else:
        # All steps completed, summarize and provide possible diagnosis
        summary = "\n".join([f"{k.replace('_',' ')}: {v}" for k, v in st.session_state.doctor_answers.items()])
        st.markdown("### Summary of your answers:")
        st.markdown(summary)
        st.info("Generating possible diagnosis and suggested next steps...")

        # Compose prompt for Gemini
        diagnostic_prompt = (
            "You are a careful, expert medical AI. Given the following patient history, "
            "please do the following:\n"
            "1. Summarize the key findings.\n"
            "2. List the most likely differential diagnoses (with reasoning).\n"
            "3. Suggest the most appropriate next diagnostic tests (with justification).\n"
            "4. Suggest a general plan for management and follow-up.\n"
            "5. Remind the user that this is not a real diagnosis and they must consult a healthcare provider.\n\n"
            f"Patient history:\n{summary}"
        )
        with st.spinner("Doctor is thinking..."):
            response = get_gemini_response([{"role": "user", "content": diagnostic_prompt}])
        st.markdown(f"**Doctor:** {response}")

        # Option to restart
        if st.button("Start new analysis"):
            st.session_state.doctor_step = 0
            st.session_state.doctor_answers = {}
            st.experimental_rerun()

    st.caption("Disclaimer: This is a simulated assistant for informational purposes only. Always consult a licensed healthcare provider.")

# --- REPORT RESULT MODE ---

elif mode == "Report Result":
    st.header("📄 Report Result")
    st.info("Upload a **medical report** (PDF, JPG, PNG). The AI will attempt to analyze and summarize it. Non-medical documents will not be processed.")

    uploaded_file = st.file_uploader("Upload your medical report (PDF, JPG, PNG):", type=["pdf", "jpg", "jpeg", "png"])

    if uploaded_file:
        file_type = uploaded_file.type.split("/")[-1]
        file_bytes = uploaded_file.read()
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
