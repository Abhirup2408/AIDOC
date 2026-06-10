import streamlit as st
import google.generativeai as genai
import os

# --- CONFIGURATION ---
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") 
if not GEMINI_API_KEY:
    st.error("Please set your GEMINI_API_KEY environment variable or Streamlit secret.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-3.5-flash"  

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

def get_gemini_response(messages, model=MODEL_NAME, vision=False, file_data=None, file_type=None, system_instruction=None):
   
    model_instance = genai.GenerativeModel(model_name=model, system_instruction=system_instruction)
    
    if vision and file_data:
        parts = [{"text": messages[-1]["content"]}]
        mime_type = f"image/{file_type}" if file_type in ["jpg", "jpeg", "png"] else "application/pdf"
        parts.append({"mime_type": mime_type, "data": file_data})
        
        response = model_instance.generate_content(parts)
        return response.text
    else:
       
        formatted_history = []
   
        for msg in messages[:-1]:
            formatted_history.append({
                "role": "user" if msg["role"] == "user" else "model",
                "parts": [msg["content"]]
            })
            
        chat = model_instance.start_chat(history=formatted_history)
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

    # Display chat history container
    for chat_msg in st.session_state.student_history:
        with st.chat_message(chat_msg["role"]):
            st.markdown(chat_msg["content"])

    # Capturing input using a chat input interface instead of a regular text input
    if query := st.chat_input("Enter your medical question:"):
        if not is_medical_query(query):
            st.warning("Please ask only medical-related questions. Non-medical queries are not supported.")
        else:
            # Display user message instantly
            with st.chat_message("user"):
                st.markdown(query)
            st.session_state.student_history.append({"role": "user", "content": query})
            
            system_prompt = (
                "You are a helpful medical assistant. Only answer medical, health, or doctor-related questions. "
                "If the query is not medical, politely refuse to answer."
            )
            
            with st.spinner("Thinking..."):
                try:
                    response = get_gemini_response(
                        messages=st.session_state.student_history,
                        system_instruction=system_prompt
                    )
                    with st.chat_message("model"):
                        st.markdown(response)
                    st.session_state.student_history.append({"role": "model", "content": response})
                except Exception as e:
                    st.error(f"API Error: {e}")

# --- DOCTOR ANALYSIS MODE ---
elif mode == "Doctor Analysis":
    st.header("👨‍⚕️ Doctor Analysis")
    st.info("Simulated doctor: Step-by-step medical history and possible diagnosis.")

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

    # Display past logs cleanly
    for idx in range(current_step):
        prev_name, _ = clinical_steps[idx]
        prev_answer = st.session_state.doctor_answers.get(prev_name, "")
        st.markdown(f"**{prev_name.replace('_', ' ')}:** {prev_answer}")

    if current_step < len(clinical_steps):
        step_name, step_question = clinical_steps[current_step]
        st.markdown(f"--- \n### Current Step: *{step_name.replace('_', ' ')}*")
        
        # Use a form to prevent random trigger fires on keystrokes
        with st.form(key=f"form_{current_step}"):
            user_input = st.text_input(step_question)
            submit_button = st.form_submit_button(label="Submit")
            
            if submit_button and user_input:
                if current_step == 0 and not is_medical_query(user_input):
                    st.warning("Please answer with a valid medical concern.")
                else:
                    st.session_state.doctor_answers[step_name] = user_input
                    st.session_state.doctor_step += 1
                    st.rerun()
    else:
        summary = "\n".join([f"{k.replace('_',' ')}: {v}" for k, v in st.session_state.doctor_answers.items()])
        st.markdown("### Summary of your answers:")
        st.text(summary)
        st.info("Generating possible diagnosis and suggested next steps...")

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
            try:
                response = get_gemini_response([{"role": "user", "content": diagnostic_prompt}])
                st.markdown(f"**Doctor:**\n{response}")
            except Exception as e:
                st.error(f"API Error: {e}")

        if st.button("Start new analysis"):
            st.session_state.doctor_step = 0
            st.session_state.doctor_answers = {}
            st.rerun()

    st.caption("Disclaimer: This is a simulated assistant for informational purposes only.")

# --- REPORT RESULT MODE ---
elif mode == "Report Result":
    st.header("📄 Report Result")
    st.info("Upload a medical report (PDF, JPG, PNG).")

    uploaded_file = st.file_uploader("Upload your medical report:", type=["pdf", "jpg", "jpeg", "png"])

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

else:
    st.markdown("""
    ### Welcome to the AI Medical Assistant!
    - **Student Help:** Ask medical questions and get clear answers.
    - **Doctor Analysis:** Simulated doctor will ask you step-by-step history and provide possible insights.
    - **Report Result:** Upload a medical report for AI analysis.
    """)
