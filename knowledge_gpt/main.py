import streamlit as st
import pandas as pd
import io
import xml.etree.ElementTree as ET
import openai


from knowledge_gpt.components.sidebar import sidebar

from knowledge_gpt.ui import (
    wrap_doc_in_html,
    is_query_valid,
    is_file_valid,
    is_open_ai_key_valid,
    display_file_read_error,
)

from knowledge_gpt.core.caching import bootstrap_caching

from knowledge_gpt.core.parsing import read_file
from knowledge_gpt.core.chunking import chunk_file
from knowledge_gpt.core.embedding import embed_files
from knowledge_gpt.core.qa import query_folder
from knowledge_gpt.core.qa import query_single_prompt
from knowledge_gpt.core.utils import get_llm
from langchain.llms import OpenAI


EMBEDDING = "openai"
VECTOR_STORE = "faiss"
MODEL_LIST = ["gpt-3.5-turbo", "gpt-4"]

# Uncomment to enable debug mode
# MODEL_LIST.insert(0, "debug")

st.set_page_config(page_title="LegalDocsGPT", page_icon="📖", layout="wide")
st.header("📖LegalDocsGPT")

# Enable caching for expensive functions
bootstrap_caching()

sidebar()

openai_api_key = st.session_state.get("OPENAI_API_KEY")


if not openai_api_key:
    st.warning(
        "Enter your OpenAI API key in the sidebar. You can get a key at"
        " https://platform.openai.com/account/api-keys."
    )



# uploaded_file = st.file_uploader(
#     "Upload a pdf, docx, or txt file",
#     type=["pdf", "docx", "txt"],
#     help="Scanned documents are not supported yet!",
# )

# Change the file uploader to accept xml file
uploaded_file = st.file_uploader(
    "Upload law as xml",
    type=["xml"],
    accept_multiple_files=False,
    help="Scanned documents are not supported yet!",
)

# Input voor bedrijfsprofiel
bedrijfsprofiel = st.text_area("Voer Bedrijfsprofiel in", "Beschrijf hier uw bedrijfsprofiel...")

# def extract_articles_from_xml(file_path):
#     tree = ET.parse(file_path)
#     root = tree.getroot()

#     articles = []
#     for article in root.findall(".//artikel"):
#         article_data = {'label': article.get('label', 'N/A'),
#                         'inwerking': article.get('inwerking', 'N/A'),
#                         'status': article.get('status', 'N/A')}
#         articles.append(article_data)
#     return articles

def extract_articles_from_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    articles = []
    for article in root.findall(".//artikel"):
        if article.get('status') != 'vervallen':
            chapter = article.get('bwb-ng-variabel-deel', 'Onbekend hoofdstuk').split('/')[2]  # Extracting chapter
            article_data = {'label': article.get('label', 'N/A'),
                            'chapter': chapter,
                            'inwerking': article.get('inwerking', 'N/A')}
            articles.append(article_data)
    return articles

def load_articles(filepath):
    return extract_articles_from_xml(filepath)

model: str = st.selectbox("Model", options=MODEL_LIST)  # type: ignore

with st.expander("Advanced Options"):
    return_all_chunks = st.checkbox("Show all chunks retrieved from vector search")
    show_full_doc = st.checkbox("Show parsed contents of the document")

if not uploaded_file:
    st.stop()

if not is_open_ai_key_valid(openai_api_key, model):
    st.stop()

loaded_articles = load_articles(uploaded_file)

selected_articles = {}
for article in loaded_articles:
    st.subheader(f"Hoofdstuk: {article['chapter']}")
    if st.checkbox(f"{article['label']} (Inwerking: {article['inwerking']})"):
        selected_articles[article['label']] = article

folder_indices = []

# LLM Prompt genereren en queryen
if st.button('Voer LLM Query uit voor Geselecteerde Artikelen'):
    # Initialiseer LLM
    # Setting the API key
    openai.api_key = openai_api_key

    for label, article_data in selected_articles.items():
        prompt = f"Welke compliance verplichtingen vloeien voort uit dit artikel voor het gegeven bedrijfsprofiel? {article_data['label']} {bedrijfsprofiel}"
        
        completion = openai.ChatCompletion.create(
          # Use GPT 3.5 as the LLM
          model=model,
          # Pre-define conversation messages for the possible roles
          messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
          ]
        )
        result = completion.choices[0].message

        # Resultaten weergeven
        st.write(f"Resultaat voor {label}:")
        st.write(result.answer)  # Of een andere methode om de resultaten te tonen


# for uploaded_file in uploaded_files:
#     try:
#         file = read_file(uploaded_file)
#     except Exception as e:
#         display_file_read_error(e, file_name=uploaded_file.name)
#         continue  # Skip the current file and continue with the next

#     if not is_file_valid(file):
#         continue  # Skip the current file and continue with the next

#     chunked_file = chunk_file(file, chunk_size=300, chunk_overlap=0)

#     with st.spinner(f"Indexing {uploaded_file.name}... This may take a while⏳"):
#         folder_index = embed_files(
#             files=[chunked_file],
#             embedding=EMBEDDING if model != "debug" else "debug",
#             vector_store=VECTOR_STORE if model != "debug" else "debug",
#             openai_api_key=openai_api_key,
#         )
#         folder_indices.append((uploaded_file.name, folder_index))

# if not uploaded_file:
#     st.stop()

# try:
#     file = read_file(uploaded_file)
# except Exception as e:
#     display_file_read_error(e, file_name=uploaded_file.name)

# chunked_file = chunk_file(file, chunk_size=300, chunk_overlap=0)

# if not is_file_valid(file):
#     st.stop()





# with st.spinner("Indexing document... This may take a while⏳"):
#     folder_index = embed_files(
#         files=[chunked_file],
#         embedding=EMBEDDING if model != "debug" else "debug",
#         vector_store=VECTOR_STORE if model != "debug" else "debug",
#         openai_api_key=openai_api_key,
#     )

# # Initialize a session state for the number of questions
# if 'num_questions' not in st.session_state:
#     st.session_state['num_questions'] = 1

# # Function to add a question
# def add_question():
#     st.session_state['num_questions'] += 1

# # Function to remove a question
# def remove_question():
#     st.session_state['num_questions'] = max(1, st.session_state['num_questions'] - 1)

# # Add a button to add more questions
# with st.form(key="questions_form"):
#     for i in range(st.session_state['num_questions']):
#         st.text_area(f"Question {i+1}", key=f"question_{i}")
#     submitted = st.form_submit_button("Submit")

# # Add buttons to add/remove question fields
# st.button("Add another question", on_click=add_question)
# st.button("Remove last question", on_click=remove_question)

# if show_full_doc:
#     with st.expander("Document"):
#         # Hack to get around st.markdown rendering LaTeX
#         st.markdown(f"<p>{wrap_doc_in_html(file.docs)}</p>", unsafe_allow_html=True)


# if submit:
#     if not is_query_valid(query):
#         st.stop()

#     # Output Columns
#     answer_col, sources_col = st.columns(2)

#     llm = get_llm(model=model, openai_api_key=openai_api_key, temperature=0)

#     for file_name, folder_index in folder_indices:
#         result = query_folder(
#             folder_index=folder_index,
#             query=query,
#             return_all=return_all_chunks,
#             llm=llm,
#         )

#         with answer_col:
#             st.markdown(f"#### Answer for {file_name}")
#             st.markdown(result.answer)

#         with sources_col:
#             st.markdown(f"#### Sources for {file_name}")
#             for source in result.sources:
#                 st.markdown(source.page_content)
#                 st.markdown(source.metadata["source"])
#                 st.markdown("---")

# # Process the files when questions are submitted
# if submitted:
#     # Initialize a list to store results for Excel
#     excel_data = []

#     # Iterate through all questions
#     for i in range(st.session_state['num_questions']):
#         query = st.session_state[f"question_{i}"]
#         if not is_query_valid(query):
#             continue

#         answer_col, sources_col = st.columns(2)

#         llm = get_llm(model=model, openai_api_key=openai_api_key, temperature=0)

#         for file_name, folder_index in folder_indices:
#             result = query_folder(
#                 folder_index=folder_index,
#                 query=query,
#                 return_all=return_all_chunks,
#                 llm=llm,
#             )

#             # Prepare data for the Excel file
#             row = [file_name, f"Question {i+1}", query, result.answer]
#             for source in result.sources:
#                 row.extend([source.page_content, source.metadata.get("source", "")])

#             excel_data.append(row)

#             # Display the answers and sources
#             with answer_col:
#                 st.markdown(f"#### Answer for {file_name} - Question {i+1}")
#                 st.markdown(result.answer)

#             with sources_col:
#                 st.markdown(f"#### Sources for {file_name} - Question {i+1}")
#                 for source in result.sources:
#                     st.markdown(source.page_content)
#                     st.markdown(source.metadata["source"])
#                     st.markdown("---")

#     # Determine the maximum number of sources
#     max_sources = max((len(row) - 4) // 2 for row in excel_data)

#     # Dynamically create column names
#     column_names = ["Document Name", "Question ID", "Question Text", "Answer"]
#     for i in range(1, max_sources + 1):
#         column_names.extend([f"Source {i}", f"Page Nr {i}"])

#     # Create a DataFrame with dynamic columns
#     df = pd.DataFrame(excel_data, columns=column_names)

#     # Excel file creation and download button
#     excel_file = io.BytesIO()
#     with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
#         df.to_excel(writer, index=False, sheet_name="Results")
#         writer.close()
#     excel_file.seek(0)

#     st.download_button(
#         label="Download Results as Excel",
#         data=excel_file,
#         file_name="query_results.xlsx",
#         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#     )

