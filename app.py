import streamlit as st
import os
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from supabase import create_client, Client
import tempfile
import pandas as pd

# ── Load environment variables ────────────────────────────────
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="OAF Knowledge Base Manager",
    page_icon="🌱",
    layout="wide"
)

# ════════════════════════════════════════════════════════════════
# AUTHENTICATION
# ════════════════════════════════════════════════════════════════

ROLE_PERMISSIONS = {
    "admin":    {"upload": True,  "delete": True,  "view": True, "search": True},
    "agronomy": {"upload": True,  "delete": False, "view": True, "search": True},
    "viewer":   {"upload": False, "delete": False, "view": True, "search": True},
}

ROLE_LABELS = {
    "admin":    "🔴 Admin",
    "agronomy": "🟢 Agronomy Supervisor",
    "viewer":   "🔵 Regional Lead / Viewer",
}

def check_login():
    if "user_role" not in st.session_state:
        st.session_state.user_role = None
        st.session_state.username  = None

    if st.session_state.user_role is None:
        st.title("🌱 OAF Uganda — Knowledge Base Manager")
        st.markdown("Please log in to continue.")
        st.divider()

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.form("login_form"):
                st.subheader("🔐 Login")
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submit   = st.form_submit_button("Login", use_container_width=True, type="primary")

                if submit:
                    users = st.secrets.get("users", {})
                    if username in users and users[username]["password"] == password:
                        st.session_state.user_role = users[username]["role"]
                        st.session_state.username  = username
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password.")

            st.caption("Contact your Data Lead if you need access credentials.")
        return False
    return True

def can(action):
    role = st.session_state.get("user_role", "viewer")
    return ROLE_PERMISSIONS.get(role, {}).get(action, False)

# ── Stop here if not logged in ────────────────────────────────
if not check_login():
    st.stop()

# ════════════════════════════════════════════════════════════════
# AUTHENTICATED — Initialize Clients
# ════════════════════════════════════════════════════════════════

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_resource
def init_embeddings():
    with st.spinner("Loading embedding model (first time only, ~30 seconds)..."):
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

supabase   = init_supabase()
embeddings = init_embeddings()

# ── Header ────────────────────────────────────────────────────
st.title("🌱 OAF Uganda — Agronomy Knowledge Base Manager")
st.markdown("Upload OAF agronomy documents to power the AI advisory system for field officers.")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 👤 Logged In As")
    st.markdown(f"**{st.session_state.username}**")
    st.markdown(f"{ROLE_LABELS.get(st.session_state.user_role, '')}")

    perms = ROLE_PERMISSIONS.get(st.session_state.user_role, {})
    st.markdown("**Your permissions:**")
    st.markdown(f"- Upload documents: {'✅' if perms.get('upload') else '❌'}")
    st.markdown(f"- Delete documents: {'✅' if perms.get('delete') else '❌'}")
    st.markdown(f"- View knowledge base: {'✅' if perms.get('view') else '❌'}")
    st.markdown(f"- Test search: {'✅' if perms.get('search') else '❌'}")

    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.user_role = None
        st.session_state.username  = None
        st.rerun()

    st.divider()
    st.header("📋 Document Settings")

    species = st.selectbox(
        "Tree Species",
        ["All Species", "Grevillea robusta", "Faidherbia albida", "Albizia coriaria"]
    )

    topic = st.selectbox(
        "Topic",
        [
            "General Production",
            "Seed Sowing & Germination",
            "Pricking & Transplanting",
            "Watering & Irrigation",
            "Pest & Disease Management",
            "Fertilizer & Nutrition",
            "Shade Management",
            "Hardening Off",
            "Height & Growth",
            "Raised Bed Construction",
            "Nursery Setup",
        ]
    )

    source = st.text_input(
        "Document Source",
        placeholder="e.g. OAF Grevillea Guide 2026"
    )

    chunk_size = st.slider(
        "Chunk Size (words)",
        min_value=200, max_value=1000, value=500, step=50,
        help="Smaller chunks = more precise answers."
    )

    chunk_overlap = st.slider(
        "Chunk Overlap (words)",
        min_value=0, max_value=200, value=50, step=10,
        help="Overlap between chunks to avoid cutting mid-topic."
    )

    st.divider()
    st.markdown("**Connected to:**")
    st.markdown(f"🗄️ Supabase: `{SUPABASE_URL[:30]}...`")
    st.markdown("🤖 Model: `all-MiniLM-L6-v2` (free)")

# ── Build tabs based on role permissions ─────────────────────
tab_labels = []
if can("upload"):
    tab_labels.append("📤 Upload Document")
if can("view"):
    tab_labels.append("📚 View Knowledge Base")
if can("search"):
    tab_labels.append("🔍 Test Search")

tabs      = st.tabs(tab_labels)
tab_index = 0

# ════════════════════════════════════════════════════════════════
# TAB — Upload Document (admin + agronomy only)
# ════════════════════════════════════════════════════════════════
if can("upload"):
    with tabs[tab_index]:
        st.subheader("Upload Agronomy Document")

        uploaded_file = st.file_uploader(
            "Drag and drop your document here",
            type=["pdf", "docx"],
            help="Supported formats: PDF (.pdf) and Word Document (.docx)"
        )

        # Unsupported format check
        if uploaded_file is not None:
            file_ext = uploaded_file.name.split(".")[-1].lower()
            if file_ext not in ["pdf", "docx"]:
                st.error(
                    f"❌ Format .{file_ext} is not supported. "
                    f"Please upload a PDF or Word document (.docx)."
                )
                uploaded_file = None

        if uploaded_file is not None:
            file_ext  = uploaded_file.name.split(".")[-1].lower()
            file_type = "PDF" if file_ext == "pdf" else "Word Document"
            st.success(
                f"✅ File loaded: **{uploaded_file.name}** "
                f"({file_type} — {round(uploaded_file.size/1024, 1)} KB)"
            )

            if not source:
                st.warning("⚠️ Please enter a Document Source in the sidebar before uploading.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    preview_btn = st.button("👁️ Preview Chunks", use_container_width=True)
                with col2:
                    upload_btn = st.button(
                        "🚀 Upload to Knowledge Base",
                        use_container_width=True,
                        type="primary"
                    )

                if preview_btn or upload_btn:
                    with st.spinner("Extracting text from document..."):
                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=f".{file_ext}"
                        ) as tmp:
                            tmp.write(uploaded_file.read())
                            tmp_path = tmp.name

                        # Load based on file type
                        if file_ext == "pdf":
                            loader = PyPDFLoader(tmp_path)
                        elif file_ext == "docx":
                            loader = Docx2txtLoader(tmp_path)
                        else:
                            st.error("❌ Document format not supported.")
                            os.unlink(tmp_path)
                            st.stop()

                        pages    = loader.load()
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            separators=["\n\n", "\n", ".", " "]
                        )
                        chunks = splitter.split_documents(pages)
                        os.unlink(tmp_path)

                    st.info(
                        f"📄 **{len(pages)} pages** extracted → "
                        f"split into **{len(chunks)} chunks**"
                    )

                    # ── Preview ───────────────────────────────
                    if preview_btn:
                        st.subheader(f"Preview — First 3 Chunks of {len(chunks)}")
                        for i, chunk in enumerate(chunks[:3]):
                            with st.expander(
                                f"Chunk {i+1} — {len(chunk.page_content.split())} words"
                            ):
                                st.write(chunk.page_content)
                                st.caption(f"Page: {chunk.metadata.get('page', 'N/A')}")

                    # ── Upload ────────────────────────────────
                    if upload_btn:
                        metadata = {
                            "source":      source,
                            "species":     species,
                            "topic":       topic,
                            "filename":    uploaded_file.name,
                            "pages":       len(pages),
                            "uploaded_by": st.session_state.username,
                        }

                        progress_bar  = st.progress(0)
                        status_text   = st.empty()
                        success_count = 0
                        error_count   = 0

                        for i, chunk in enumerate(chunks):
                            try:
                                status_text.text(
                                    f"Embedding chunk {i+1} of {len(chunks)}..."
                                )
                                vector     = embeddings.embed_query(chunk.page_content)
                                chunk_meta = {
                                    **metadata,
                                    "page":        chunk.metadata.get("page", 0),
                                    "chunk_index": i,
                                }
                                supabase.table("oaf_knowledge_base").insert({
                                    "content":   chunk.page_content,
                                    "embedding": vector,
                                    "metadata":  chunk_meta,
                                }).execute()
                                success_count += 1
                            except Exception as e:
                                error_count += 1
                                st.error(f"Error on chunk {i+1}: {str(e)}")
                            progress_bar.progress((i + 1) / len(chunks))

                        status_text.empty()

                        if success_count > 0:
                            st.success(f"""
                            ✅ **Upload Complete!**
                            - Document: {uploaded_file.name}
                            - Chunks stored: {success_count}
                            - Errors: {error_count}
                            - Species: {species}
                            - Topic: {topic}
                            - Uploaded by: {st.session_state.username}
                            """)

                        if error_count > 0:
                            st.warning(
                                f"⚠️ {error_count} chunks failed. "
                                f"Check your Supabase connection."
                            )

    tab_index += 1

# ════════════════════════════════════════════════════════════════
# TAB — View Knowledge Base (all roles)
# ════════════════════════════════════════════════════════════════
if can("view"):
    with tabs[tab_index]:
        st.subheader("Knowledge Base Contents")

        col1, col2 = st.columns([3, 1])
        with col2:
            st.button("🔄 Refresh", use_container_width=True)

        try:
            response = supabase.table("oaf_knowledge_base")\
                .select("id, content, metadata, created_at")\
                .order("created_at", desc=True)\
                .execute()

            records = response.data

            if not records:
                st.info("No documents uploaded yet. Go to the Upload tab to add your first document.")
            else:
                total_chunks = len(records)
                sources = list(set([
                    r["metadata"].get("source", "Unknown")
                    for r in records if r.get("metadata")
                ]))
                species_list = list(set([
                    r["metadata"].get("species", "Unknown")
                    for r in records if r.get("metadata")
                ]))

                m1, m2, m3 = st.columns(3)
                m1.metric("Total Chunks", total_chunks)
                m2.metric("Documents", len(sources))
                m3.metric("Species Covered", len(species_list))
                st.divider()

                st.subheader("Documents in Knowledge Base")
                for source_name in sources:
                    source_chunks = [
                        r for r in records
                        if r.get("metadata", {}).get("source") == source_name
                    ]

                    with st.expander(
                        f"📄 {source_name} — {len(source_chunks)} chunks"
                    ):
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            if source_chunks[0].get("metadata"):
                                meta = source_chunks[0]["metadata"]
                                st.markdown(f"**Species:** {meta.get('species', 'N/A')}")
                                st.markdown(f"**Topic:** {meta.get('topic', 'N/A')}")
                                st.markdown(f"**Pages:** {meta.get('pages', 'N/A')}")
                                st.markdown(
                                    f"**Uploaded by:** {meta.get('uploaded_by', 'N/A')}"
                                )

                        with col2:
                            if can("delete"):
                                if st.button("🗑️ Delete", key=f"del_{source_name}"):
                                    ids = [r["id"] for r in source_chunks]
                                    for doc_id in ids:
                                        supabase.table("oaf_knowledge_base")\
                                            .delete()\
                                            .eq("id", doc_id)\
                                            .execute()
                                    st.success(f"Deleted {len(ids)} chunks")
                                    st.rerun()
                            else:
                                st.caption("🔒 Delete: Admin only")

                        st.markdown("**Sample chunk:**")
                        st.caption(source_chunks[0]["content"][:300] + "...")

        except Exception as e:
            st.error(f"Error fetching knowledge base: {str(e)}")
            st.info(
                "Make sure your Supabase credentials are correct "
                "and the oaf_knowledge_base table exists."
            )

    tab_index += 1

# ════════════════════════════════════════════════════════════════
# TAB — Test Search (all roles)
# ════════════════════════════════════════════════════════════════
if can("search"):
    with tabs[tab_index]:
        st.subheader("Test Semantic Search")
        st.markdown(
            "Test how the knowledge base responds to field officer questions. "
            "This is exactly how the Telegram advisory bot searches for answers."
        )

        test_query = st.text_input(
            "Enter a test question",
            placeholder="e.g. My grevillea seedlings are turning yellow, what should I do?"
        )

        col1, col2 = st.columns(2)
        with col1:
            filter_species = st.selectbox(
                "Filter by species (optional)",
                ["All", "Grevillea robusta", "Faidherbia albida", "Albizia coriaria"],
                key="search_species"
            )
        with col2:
            top_k = st.slider("Number of results to retrieve", 1, 10, 3)

        similarity_threshold = st.slider(
            "Similarity threshold",
            min_value=0.1, max_value=0.9, value=0.5, step=0.05,
            help="Higher = only very close matches. Lower = broader results."
        )

        if st.button("🔍 Search Knowledge Base", type="primary") and test_query:
            with st.spinner("Searching knowledge base..."):
                try:
                    # Convert question to vector
                    query_vector = embeddings.embed_query(test_query)

                    # Search Supabase pgvector
                    response = supabase.rpc(
                        "match_oaf_documents",
                        {
                            "query_embedding": query_vector,
                            "match_threshold":  similarity_threshold,
                            "match_count":      top_k
                        }
                    ).execute()

                    results = response.data

                    if not results:
                        st.warning(
                            "No matching documents found. "
                            "Try lowering the similarity threshold or uploading more documents."
                        )
                    else:
                        st.success(f"✅ Found {len(results)} relevant chunks")
                        st.divider()

                        for i, result in enumerate(results):
                            similarity = round(result.get("similarity", 0), 3)
                            meta       = result.get("metadata", {})

                            with st.expander(
                                f"Result {i+1} — Similarity: {similarity} | "
                                f"{meta.get('source', 'Unknown source')}"
                            ):
                                st.write(result.get("content", ""))
                                st.divider()
                                col1, col2, col3 = st.columns(3)
                                col1.caption(f"📗 Species: {meta.get('species', 'N/A')}")
                                col2.caption(f"📌 Topic: {meta.get('topic', 'N/A')}")
                                col3.caption(f"📄 Page: {meta.get('page', 'N/A')}")

                except Exception as e:
                    st.error(f"Search error: {str(e)}")
                    st.info(
                        "Make sure you have run the match_oaf_documents "
                        "SQL function in Supabase SQL Editor."
                    )

# ── Footer ─────────────────────────────────────────────────────
st.divider()
st.caption(
    f"OAF Uganda AI Advisory System — Knowledge Base Manager v1.0 | "
    f"Logged in as: {st.session_state.username} "
    f"({st.session_state.user_role}) | "
    f"Built with Streamlit + LangChain + Supabase pgvector"
)
