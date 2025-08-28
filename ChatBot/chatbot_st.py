
from csb import get_embedder, get_generator, build_or_load_index, clean_text, chunk_text, read_any, add_texts_to_index, persist_index, retriever, synthesize_answer
import streamlit as st
import os
from typing import List,Tuple,Dict,Any

#streamlit app
st.set_page_config(page_title="AI Support Assistant (Multi-doc, Local)",page_icon="🤖",layout="wide")
st.title("🤖 AI-Powered Customer Support Assistant")
st.caption("Runs locally • No API keys • Multi-document RAG • Citations • Session memory")

#side: model and index controls
with st.sidebar:
    st.subheader("Knowledge Base")
    st.write("Upload PDFs,DOCX,TXT,CSV. They'll be embedded locally and indexed.")

    uploded_files = st.file_uploader(
        "Add documents",
        type=["pdf","docx","txt","csv"],
        accept_multiple_files=True
    )
    paste_text = st.text_area("Or paste text (optional):",height=120,placeholder="Paste product manuals,FAQ,SOPs...")

    chunk_size = st.slider("Chunk size(char)",400,1500,800,50)
    overlap = st.slider("Overlap (char)",50,400,120,10)
    top_k = st.slider("Citations (top_k)",1,5,3,1)

    #load components
    embedder = get_embedder()
    generator = get_generator()
    index,metadata = build_or_load_index(embedder)

    if st.button("Ingest now",use_container_width=True):
        new_texts = []

        #uploaded files
        for f in uploded_files or []:
            try:
                ftype,raw = read_any(f)
                chunks = chunk_text(raw,chunk_size=chunk_size,overlap=overlap)
                new_texts.extend([(f.name,c) for c in chunks])
            except Exception as e:
                st.error(f"Failed to read {f.name}:{e}")
        
        #pasted files
        if paste_text.strip():
            chunks = chunk_text(paste_text,chunk_size=chunk_size,overlap=overlap)
            new_texts.extend([("Pasted text", c) for c in chunks])

        if new_texts:
            by_source:Dict[str,List[str]] = {}
            for src, c in new_texts:
                by_source.setdefault(src,[]).append(c)
            
            total_added = 0
            for src, chunks in by_source.items():
                add_texts_to_index(chunks, src, embedder, index, metadata)
                total_added += len(chunks)

            persist_index(index,metadata)
            st.success(f"Ingested {total_added} chunks from {len(by_source)} sources")

        else:
            st.info("No new content to ingest.")
    
    if st.button("Reset Index",type = "secondary",use_container_width = True):
        try:
            os.remove(os.path.join("storage","faiss.index"))
        except FileNotFoundError:
            pass
        try:
            os.remove(os.path.join("storage","meta.npy"))
        except FileNotFoundError:
            pass
        st.experimental_rerun()
    
    st.markdown("---")
    st.write("***Index size:***",getattr(index,"ntotal",0))
    if os.path.exists("data/faqs.txt"):
        st.write("Tip: add your starter FAQs in `data/faqs.txt` and ingest them too.")

#coversational state
if "history" not in st.session_state:
    st.session_state.history = []

# Show chat history
for msg in st.session_state.history:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Chat input
user_query = st.chat_input("Ask about your product/service, refunds, onboarding, SLAs, etc.")
if user_query:
    st.session_state.history.append({"role": "user", "content": user_query})
    st.chat_message("user").markdown(user_query)

    # Retrieve
    contexts = retriever(user_query, embedder, index, metadata, k=top_k)

    # Generate grounded answer
    answer = synthesize_answer(user_query, contexts, st.session_state.history, generator)
    st.session_state.history.append({"role": "assistant", "content": answer})

    # Display answer
    st.chat_message("assistant").markdown(answer)

    # Show citations panel
    if contexts:
        with st.expander("🔎 Sources"):
            for i, c in enumerate(contexts, 1):
                st.markdown(f"**[{i}] {c['source']} — chunk {c['chunk_id']}**")
                st.markdown(f"> {c['text'][:400]}{'…' if len(c['text'])>400 else ''}")
    else:
        st.info("No sources found yet. Ingest documents in the left panel.")


