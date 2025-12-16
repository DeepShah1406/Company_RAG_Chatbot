# Chaneg "Company Name" to the actual company name you are building the chatbot for.
# Throughout the code, replace "Company Name" with the actual name.
# This is a FastAPI backend combined with a Streamlit frontend for a chatbot using LangChain and Groq LLM.
# You should have a config.json file in the same directory with your GROQ_API_KEY.
# Make sure to install all required packages:
# pip install requirements.txt and the requirements.txt should include:
# fastapi, uvicorn, pydantic, streamlit, langchain, langchain-huggingface, langchain-chroma, langchain-groq, fpdf
# To run the project:
# - Run streamlit: streamlit run main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import json
import re
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from fpdf import FPDF  # Using the original fpdf package
from datetime import datetime

def remove_emojis(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

working_dir = os.path.dirname(os.path.realpath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))
GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# FastAPI app and Pydantic model for API input
app = FastAPI()

class MessageRequest(BaseModel):
    message: str

# FastAPI route for chatbot
@app.post("/chat")
async def chatbot(request: MessageRequest):
    message = request.message

    # Setup vectorstore (same as Streamlit code)
    vectorstore = setup_vectorstore()

    # Setup the conversational chain (same as Streamlit code)
    conversational_chain = chat_chain(vectorstore)

    # Check for sensitive topics
    if contains_sensitive_topics(message):
        response = "It seems you may be asking questions outside my context, please ask questions related to Company Name only."
    else:
        # Get response from the conversational chain
        response = conversational_chain({"question": message})["answer"]

    return {"response": response}

# Default prompts
DEFAULT_SYSTEM_PROMPT = """You are a specialized AI assistant dedicated exclusively to Company Name and its services, try to keep the responsed as short as possibles.
Your primary goals are:
1. Understand the user's needs through a short onboarding questionnaire.
2. Provide accurate, concise, and helpful information strictly based on Company Name’s verified data.
3. Offer additional relevant Company Name services as suggestions when appropriate.
4. Maintain a warm, professional, and empathetic tone.

==========================
   INTERACTION GUIDELINES
==========================

# PHASE 1 — USER NEEDS INTAKE (ALWAYS DO THIS FIRST)
Before providing any information, ask the user 2–4 short onboarding questions such as:
- “What service are you looking for today?”  
- “What type of business do you run?”  
- “Do you already have a website or are you starting fresh?”  
- “What is your main goal — growth, branding, leads, or something else?”

Collect enough information to understand what Company Name services match their needs.
Do NOT skip this step unless the user already answered these questions in earlier messages.

# PHASE 2 — RESPOND USING USER DATA + Company Name DATA
Once intake data is collected:
- Use the user's answers + the vector database Company Name data.
- Provide clear, short, valuable responses (easy to read in one go).
- Use short bullet points and lists when helpful.
- Add only a few emojis to keep engagement (no overload).

# PHASE 3 — RELATED SERVICE SUGGESTIONS
After giving the main answer, suggest 1–2 related Company Name services.
Example:
If the user asks for web design → suggest SEO, branding, hosting, digital marketing, etc.

Example phrasing:
“Since you're planning a new website, you may also benefit from our SEO services to improve visibility.”

=================================
   CONTEXT & RESPONSE RULES
=================================

1. First check if the provided context contains relevant Company Name information.
2. If context is empty or irrelevant:
   → Politely inform the user that you can only discuss Company Name-related topics.
3. If context contains relevant information:
   → Provide helpful responses strictly based on the data.

=================
   TONE RULES
=================

- Be empathetic and supportive.
- Keep responses concise yet informative.
- Stay factual and evidence-based.
- Use a professional but friendly tone.

"""

DEFAULT_NEGATIVE_PROMPT = """Do NOT provide any information not supported by Company Name data or system context.
Do NOT imply you are an employee, representative, or spokesperson of Company Name.
Do NOT fabricate company services, pricing, policies, or internal details.
Do NOT offer legal, financial, or unrelated professional advice.
Do NOT respond to topics outside Company Name’s scope; instead politely say the data is unavailable.
Do NOT guess confidential or internal business information.
Do NOT generate speculative or generic business advice unrelated to verified Company Name data.
Do NOT use external sources or external knowledge beyond the authorized Company Name context.
Do NOT share personal opinions or assumptions."""

def contains_sensitive_topics(question):
    sensitive_keywords = [
    ]
    
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in sensitive_keywords)

def setup_vectorstore():
    persist_directory = f"{working_dir}/vector_db_dir"
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma(persist_directory=persist_directory,
                         embedding_function=embeddings)
    return vectorstore

def chat_chain(vectorstore, system_prompt=DEFAULT_SYSTEM_PROMPT, negative_prompt=DEFAULT_NEGATIVE_PROMPT):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )
    
    # Create a combined prompt template
    prompt_template = f"""{system_prompt}

{negative_prompt}

Context (from mental health database):
{{context}}

Chat History:
{{chat_history}}

Question: {{question}}

Answer:"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "chat_history", "question"]
    )
    
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}  # Retrieve top 3 most relevant documents
    )
    memory = ConversationBufferMemory(
        llm = llm,
        output_key = "answer",
        memory_key = "chat_history",
        return_messages = True
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = retriever,
        chain_type = "stuff",
        memory = memory,
        verbose = True,
        return_source_documents = True,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    return chain

st.set_page_config(
    page_title="Chat with Company Name's Chatbot",
    page_icon="🧠",
    layout="wide",  # Changed to wide layout to accommodate sidebar
)

# Custom CSS for sidebar styling
st.markdown("""
    <style>
    div.css-textbarboxtype {
        background-color: #EEEEEE;
        border: 1px solid #DCDCDC;
        padding: 20px 20px 20px 70px;
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
    }
    
    /* Justify text for Purpose section */
    div.css-textbarboxtype:nth-of-type(3) {
        text-align: justify;
        text-justify: inter-word;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("About Bot")
    
    # About Section
    st.markdown("## Description")
    st.markdown("""
        <div class="css-textbarboxtype">
            An AI-powered chatbot designed to provide answers related to Company Name.
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## Goals")
    st.markdown("""
        <div class="css-textbarboxtype">
            - Expansion<br>
            - Optimization<br>
            - Visibility<br>
            - Engagement<br>
            - Outreach<br>
            - Scalability
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## Purpose")
    st.markdown("""
        <div class="css-textbarboxtype">
            Designed as a seamless, user-friendly entry point to Company Name'S support system, this chatbot helps users easily access accurate information without confusion or hesitation. Whether users have questions about products, services, policies, or general assistance, the chatbot provides clear explanations, reliable guidance, and context-aware responses powered by Company Name'S verified knowledge base. By simplifying interactions and delivering timely, trustworthy answers, it enhances user experience and smoothly connects them to human support representatives whenever needed.
        </div>
    """, unsafe_allow_html=True)
    
    # Values
    st.markdown("## Our Values")
    st.markdown("""
        <div class="css-textbarboxtype">
            - Attentiveness<br>
            - Creativity<br>
            - Reliability<br>
            - Transparency<br>
            - Professionalism<br>
            - Collaboration<br>
            - Innovation<br>
            - Quality<br>
            - Integrity<br>
            - Growth
        </div>
    """, unsafe_allow_html=True)
    
    # Chat History Section
    st.markdown("---")
    st.markdown("## Chat History")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history previews
    for idx, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            if st.button(f"Chat {idx//2 + 1}: {message['content'][:30]}...", key=f"history_{idx}"):
                # Load this conversation
                st.session_state.selected_chat = idx//2
    
    # PDF Export Button
    st.markdown("---")
    if st.button("Export Chat to PDF"):
        if len(st.session_state.chat_history) > 0:
            try:
                # Create PDF
                pdf = FPDF()
                pdf.add_page()
                
                # Use Arial Unicode MS font
                pdf.set_font('Arial', '', 10)
                
                # PDF Header
                pdf.set_font('Arial', 'B', 16)
                pdf.cell(0, 10, "Company Name Chatbot - Conversation History", ln=True, align='C')
                pdf.set_font('Arial', '', 12)
                pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
                pdf.ln(10)
                
                # Add conversation
                pdf.set_font('Arial', '', 10)
                for message in st.session_state.chat_history:
                    # Role header
                    pdf.set_font('Arial', 'B', 10)
                    pdf.cell(0, 10, message["role"].capitalize(), ln=True)
                    # Message content
                    pdf.set_font('Arial', '', 10)
                    pdf.multi_cell(0, 10, remove_emojis(message["content"]))
                    pdf.ln(5)
                
                # Save PDF
                filename = f"mental_health_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                pdf.output(filename)
                
                # Create download button
                with open(filename, "rb") as f:
                    st.download_button(
                        label="Download PDF",
                        data=f,
                        file_name=filename,
                        mime="application/pdf"
                    )
                
                # Clean up the file
                os.remove(filename)
                
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")
        else:
            st.warning("No chat history to export!")

# Main chat interface
# Add header image

st.title("🧠 Company Name Chatbot")

#st.image("https://img.freepik.com/premium-vector/mental-health-awareness-month-take-care-your-body-take-care-your-health-increase-awareness_758894-821.jpg?w=1380", use_column_width=True)

#st.image("D:\Final Project\PythonDemoProject3\images\may-is-mental-health-awareness-month-diversity-silhouettes-of-adults-and-children-of-different-nationalities-and-appearances-colorful-people-contour-in-flat-style-vector.jpg")

st.image("https://static.vecteezy.com/system/resources/previews/001/912/491/large_2x/set-of-scenes-business-people-meeting-with-infographics-presentation-free-vector.jpg", use_column_width=True)

#st.image("D:\Final Project\PythonDemoProject3\images\may-is-mental-health-awareness-month-diversity-silhouettes-of-adults-and-children-of-different-nationalities-and-appearances-colorful-people-contour-in-flat-style-vector-2.jpg")

#st.image("https://static.vecteezy.com/system/resources/previews/039/630/872/large_2x/may-is-mental-health-awareness-month-diversity-silhouettes-of-adults-and-children-of-different-nationalities-and-appearances-colorful-people-contour-in-flat-style-vector.jpg", use_column_width=True)
#st.image("https://static.vecteezy.com/system/resources/previews/040/941/494/non_2x/may-is-mental-health-awareness-month-banner-with-silhouettes-of-diverse-people-and-green-ribbon-women-and-men-of-different-ages-religions-and-races-design-for-info-importance-psychological-state-vector.jpg", use_column_width=True)
#st.image("https://static.vecteezy.com/system/resources/previews/038/147/547/non_2x/may-is-mental-health-awareness-month-banner-horizontal-design-with-man-women-children-old-people-silhouette-in-flat-style-informing-about-importance-of-good-state-of-mind-well-being-presentation-vector.jpg", use_column_width=True)


if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vectorstore()

if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = chat_chain(st.session_state.vectorstore)

# Display chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask a question about Company Name")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = st.session_state.conversational_chain({"question": user_input})
        assistant_response = response["answer"]
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
