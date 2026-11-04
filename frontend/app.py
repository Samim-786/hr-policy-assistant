import requests
import streamlit as st
import os

def get_api_url():
    try:
        url = st.secrets.get("API_URL")
        if url:
            return url.rstrip("/")
    except Exception:
        pass

    return os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")


API_URL = get_api_url()

st.set_page_config(page_title="HR Policy Assistant", page_icon="📚", layout="centered")
st.markdown(
    """
    <style>
    html, body, [class*="css"], .stMarkdown p, .stMarkdown li,
    .stTextInput input, .stAlert p, label, .stCaption, [data-testid="stCaptionContainer"] {
        font-size: 1.15rem !important;
    }
    h1 { font-size: 2.6rem !important; }
    h3 { font-size: 1.7rem !important; }
    .stTabs [data-baseweb="tab"] p { font-size: 1.2rem !important; }
    .stButton button p, .stFormSubmitButton button p { font-size: 1.1rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📚 HR Policy Assistant")
st.caption("Ask questions about company policies and get answers with citations.")


def call_api(path, timeout, **kwargs):
    """POST to the backend and return JSON. Shows a clean error and returns None on failure."""
    try:
        response = requests.post(f"{API_URL}{path}", timeout=timeout, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        st.error(f"Request failed: {detail}")
        return None


admin_tab, employee_tab = st.tabs(["Admin: Upload Policies", "Employee: Ask Questions"])

# ---------- Admin ----------
with admin_tab:
    st.subheader("Upload HR Policy")
    uploaded_file = st.file_uploader("Choose a Markdown or Text file", type=["md", "txt"])

    if st.button("Upload and Index", disabled=uploaded_file is None):
        with st.spinner("Uploading..."):
            upload = call_api(
                "/documents/upload",
                timeout=30,
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type or "text/plain",
                    )
                },
            )

        if upload:
            with st.spinner("Generating embeddings and indexing..."):
                index = call_api(f"/documents/{upload['stored_as']}/index", timeout=180)

            if index:
                st.success("Document uploaded and indexed!")
                with st.expander("Indexing details"):
                    st.json(index)

# ---------- Employee ----------
with employee_tab:
    st.subheader("Ask a Question")

    with st.form("question_form"):
        question = st.text_input(
            "Your question",
            placeholder="How many casual leave days can I carry forward?",
        )
        submitted = st.form_submit_button("Ask")

    if submitted:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching policies..."):
                result = call_api("/ask", timeout=120, json={"question": question})

            if result:
                st.divider()

                if result["status"] == "answered":
                    st.success(result["answer"])
                else:
                    st.warning(result["answer"])

                citations = result.get("citations", [])
                if citations:
                    with st.expander(f"Sources ({len(citations)})", expanded=True):
                        for c in citations:
                            st.markdown(f"**{c['document_name']}** · {c['section']}")
                else:
                    st.caption("No supporting citations available.")