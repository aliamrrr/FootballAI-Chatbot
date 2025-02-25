# === General Imports ===
import os
import time
import tempfile
from io import StringIO

# === Data Manipulation & Visualization ===
import pandas as pd
import matplotlib.pyplot as plt

# === Streamlit & User Interface ===
import streamlit as st
from PIL import Image

# === LangChain & AI ===
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

# === PandasAI ===
from pandasai.llm import OpenAI
from pandasai import SmartDataframe

# Initialization and Setup
def init_template():
    if 'template' not in st.session_state:
     st.session_state.template = """
        You are a data analyst providing formal, scientific responses. Follow these guidelines:
        1/ Base responses strictly on provided data
        2/ If data is unavailable, state "Insufficient information"
        3/ Include relevant statistics
        4/ Keep responses under 100 words
        5/ Use bullet points for multiple results

        User query: {question}

        Relevant data: {context}

        Structured response:
        """


# Document Processing
def process_document(file):
    if isinstance(file, pd.DataFrame):
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.csv') as tmp_file:
            file.to_csv(tmp_file.name, index=False)
            tmp_file_path = tmp_file.name
    else:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
    document = loader.load()
    return document

# Vector Store and Embedding
def query_store(query, store):
    response = store.similarity_search(query, k=3)
    return [doc.page_content for doc in response]

def init_analytical_bot(template):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    
    prompt = PromptTemplate(input_variables=["question", "context"], template=template)
    return LLMChain(llm=llm, prompt=prompt)

def init_viz_bot():
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    
    # Can be modified by the "customize tour answers" section
    template = """
    Generate Python visualization code for ANY CSV data. Rules:
    1/ Use matplotlib/seaborn
    2/ DataFrame variable: df
    3/ Include plt.savefig('plot.png')
    4/ Return ONLY code in ``` 
    5/ Add titles/labels
    6/ Handle datetime conversion

    Request: {request}
    """
    
    prompt = PromptTemplate(input_variable="request", template=template)
    return LLMChain(llm=llm, prompt=prompt)

# Main page
def main():
    init_template()
    
    # Interface setup

    st.set_page_config(
        page_title="Soccer Analytics AI",
        page_icon="⚽",
        layout="wide"
    )
    
    st.title("⚒️ Soccer Data Chatbot")
    st.markdown("""
    <style>
        .stButton button {
            border-radius: 15px;
            transition: all 0.3s ease;
        }
        .stButton button:focus {
            outline: none;
            box-shadow: 0 0 0 3px rgba(0,123,255,0.3);
        }
        .selected-column {
            border: 2px solid #4CAF50 !important;
            opacity: 0.8;
        }
        .unselected-column {
            border: 2px solid #FF5722 !important;
            opacity: 0.8;
        }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.header("⚙️ Football Analytics Setup")
    st.sidebar.image("rbfa.jfif", width=300)
    st.sidebar.markdown("---")
    
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
    uploaded_file = st.sidebar.file_uploader("Upload Match Data CSV", type="csv")
    os.environ["OPENAI_API_KEY"] = openai_key

    if not (openai_key and uploaded_file):
        st.sidebar.markdown("---")
        st.sidebar.error("📢 Upload club data & enter API key to start analysis!")
        return

    # Tabs
    tab0, tab1, tab2, tab3 = st.tabs(["🔧 Column Selector", "📈 Match Analysis", "📊 Performance Visualization","✍️ Customize Your Answers"])

    # Columns selection (Click to delete and press Prepare data)
    with tab0:
        st.header("🔧 Select Columns for Analysis")
        df = pd.read_csv(uploaded_file)
        
        if 'selected_columns' not in st.session_state:
            st.session_state.selected_columns = df.columns.tolist()
        
        cols_per_row = 5
        columns = df.columns.tolist()
        num_rows = (len(columns) + cols_per_row - 1) // cols_per_row

        for row in range(num_rows):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                idx = row * cols_per_row + col_idx
                if idx < len(columns):
                    with cols[col_idx]:
                        column = columns[idx]
                        is_selected = column in st.session_state.selected_columns
                        btn_type = "primary" if is_selected else "secondary"
                        
                        if st.button(
                            f"✓ {column}" if is_selected else f"✗ {column}",
                            key=f"col_{column}",
                            type=btn_type,
                            use_container_width=True
                        ):
                            if is_selected:
                                st.session_state.selected_columns.remove(column)
                            else:
                                st.session_state.selected_columns.append(column)
                            st.rerun()

        if st.button("🚀 Prepare Data", use_container_width=True):
            if len(st.session_state.selected_columns) == 0:
                st.error("Please select at least one column.")
            else:
                st.session_state.filtered_df = df[st.session_state.selected_columns]
                st.success("✅ Data prepared successfully! Proceed to analysis tabs.")
            
        st.subheader("Filtered Data Preview:")
        st.write(st.session_state.filtered_df.head())
    
    # Textual Chatbot (OPENAI based)
    with tab1:
        st.header("📊 Data Analysis")
        st.markdown("*Analyze football data*")

        if 'filtered_df' in st.session_state:
            document = process_document(st.session_state.filtered_df)
        else:
            document = process_document(uploaded_file)

        vector_store = FAISS.from_documents(document, OpenAIEmbeddings())

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about the data:"):
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            context_data = query_store(prompt, vector_store)
            print(st.session_state.template)
            analyst_bot = init_analytical_bot(st.session_state.template)
            response = analyst_bot.run(question=prompt, context=context_data)

            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    # Visualizations (and text) chatbot (PandasAI and OPENAI Based)
    with tab2:
        st.header("📊 Smart Data Visualization")
        st.markdown("*Explore and visualize football data through insightful charts*")

        if 'filtered_df' in st.session_state:
            df = st.session_state.filtered_df
        else:
            df = pd.read_csv(uploaded_file)

        temp_chart_dir = "temp_charts"
        os.makedirs(temp_chart_dir, exist_ok=True)

        llm = OpenAI(api_token=openai_key)
        
        smart_df = SmartDataframe(df, config={
            "llm": llm,
            "save_charts": True,
            "save_charts_path": temp_chart_dir,
            "custom_prompts": {
                "save_chart": lambda path: path
            }
        })

        if "viz_history" not in st.session_state:
            st.session_state.viz_history = []

        # Initialize context if it doesn't exist
        if "context" not in st.session_state:
            st.session_state.context = []

        # Display previous interactions (messages)
        for msg in st.session_state.viz_history:
            with st.chat_message(msg["role"]):
                if msg["type"] == "text":
                    st.markdown(msg["content"])
                elif msg["type"] == "image":
                    st.image(msg["image"], caption=msg.get("caption", "Generated Chart"))

        # Take input for new visualization request
        if viz_prompt := st.chat_input("Describe the visualization you need..."):
            # Add the new prompt to the context
            st.session_state.context.append(f"User: {viz_prompt}")
            st.session_state.viz_history.append({
                "role": "user",
                "type": "text",
                "content": viz_prompt
            })
            
            try:
                # Full context for the assistant

                full_prompt = f"{viz_prompt}\nContext: {st.session_state.context}"

                response = smart_df.chat(full_prompt)
                time.sleep(2)

                chart_files = sorted(
                    [f for f in os.listdir(temp_chart_dir) if f.endswith(".png")],
                    key=lambda x: os.path.getmtime(os.path.join(temp_chart_dir, x)),
                    reverse=True
                )

                if chart_files:
                    latest_chart = os.path.join(temp_chart_dir, chart_files[0])
                    
                    # Add response and chart to the session history
                    st.session_state.viz_history.append({
                        "role": "assistant",
                        "type": "image",
                        "image": latest_chart,
                        "caption": f"Visualization for: {viz_prompt}"
                    })
                    
                    with st.chat_message("assistant"):
                        st.image(latest_chart, caption=f"Visualization for: {viz_prompt}")
                        plt.close('all')
                else:
                    st.session_state.viz_history.append({
                        "role": "assistant",
                        "type": "text",
                        "content": str(response)
                    })
                    with st.chat_message("assistant"):
                        st.markdown(response)

                # Add the assistant's response to the context
                st.session_state.context.append(f"Assistant: {response}")

            except Exception as e:
                error_msg = f"Visualization error: {str(e)}"
                st.session_state.viz_history.append({
                    "role": "assistant",
                    "type": "text",
                    "content": error_msg
                })
                st.error(error_msg)

    with tab3:
        st.header("✍️ Customize Your Answers Template")
        
        col1, col2 = st.columns(2)

        with col1:
            # Section Template
            st.subheader("Template for Chatbot")
            template = st.text_area("Modify the Template", """
            You are a data analyst providing formal, scientific responses. Follow these guidelines:
            1/ Base responses strictly on provided data
            2/ If data is unavailable, state "Insufficient information"
            3/ Include relevant statistics
            4/ Keep responses under 100 words
            5/ Use bullet points for multiple results

            User query: {question}

            Relevant data: {context}

            Structured response:
            """, height=300)

            # template to session state
            if st.button("Save Template"):
                st.session_state.template = template
                st.success("Template saved!")
        
        with col2:
            # Section Context for Visualization
            st.subheader("Context for Smart Visualization")
            # Allow the user to add or modify context for the smart visualization
            new_context = st.text_area("Modify Context for Visualization", "", height=150)
            
            # Save context when the user clicks "Save Context"
            if st.button("Save Context"):
                if new_context:
                    if "context" not in st.session_state:
                        st.session_state.context = []
                    st.session_state.context.append(new_context)
                    st.success("Context updated!")
                else:
                    st.error("Context cannot be empty.")
            
            st.subheader("Current Context")
            st.write(st.session_state.context if st.session_state.context else "No context provided.")
     
if __name__ == "__main__":
    main()
