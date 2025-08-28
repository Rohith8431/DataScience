#pip install streamlit

import os
import io
import re
import json
import pandas as pd
import numpy as np
import time
import streamlit as st
from typing import List,Dict,Tuple,Any

from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
from pypdf import PdfReader
from docx import Document as DocxDocument

#pip install faiss-cpu

#pip install pypdf

#pip install python-docx

#!pip install sentence-transformers

#clean & chunk text
def clean_text(text:str)->str:
  text=re.sub(r"\s+"," ",text)
  return text.strip()

def chunk_text(text:str, chunk_size:int=800, overlap:int=120) -> List[str]:
  """
  Character based chunking(simple & Robust)
  Chunk size ~800 chars works well for small model like FLAN-T5
  """
  text = clean_text(text)
  chunks = []
  start = 0
  n = len(text)
  while start < n:
    end = min(start + chunk_size, n)
    chunk = text[start:end]
    chunks.append(chunk)
    start=end-overlap
    if start < 0:
      start = 0
  return chunks

def load_txt(file_bytes: bytes)->str:
  return file_bytes.decode("utf-8",errors="ignore")

def load_pdf(file_bytes: bytes)->str:
  with io.BytesIO(file_bytes) as fb:
    reader = PdfReader(fb)
    texts=[]
    for page in reader.pages:
      try:
        t=page.extract_text() or ""
      except Exception :
        t=""
      if t:
        texts.append(t)
  return "\n".join(texts)

def load_docx(file_bytes: bytes)->str:
  with io.BytesIO(file_bytes) as fb:
    doc=DocxDocument(fb)
  return "\n".join(p.text for p in doc.paragraphs)

def load_csv(file_bytes: bytes)->str:
  with io.BytesIO(file_bytes) as fb:
    df=pd.read_csv(fb)
    #converting to readable FAQ-like table text
  return df.to_csv(index=False)

def read_any(file)->Tuple[str,str]:
  name=file.name.lower()
  content=file.read()
  if name.endswith(".pdf"):
    return "pdf",load_pdf(content)
  elif name.endswith(".docx"):
    return "docx",load_docx(content)
  elif name.endswith(".csv"):
    return "csv",load_csv(content)
  elif name.endswith(".txt"):
    return "txt",load_txt(content)
  else:
    raise ValueError("Unsupported file type.Please upload PDF,DOCX,TXT, OR CSV.")

#Embedding + FAISS handling
@st.cache_resource
def get_embedder():
  return SentenceTransformer("all-MiniLM-L6-v2")

def build_or_load_index(
    embedder: SentenceTransformer,
    storage_dir: str = "storage"
) -> Tuple[faiss.IndexFlatL2, List[Dict[str, Any]]]:
    os.makedirs(storage_dir, exist_ok=True)
    index_path = os.path.join(storage_dir, "faiss.index")
    meta_path = os.path.join(storage_dir, "meta.npy")

    if os.path.exists(index_path) and os.path.exists(meta_path):
        index = faiss.read_index(index_path)
        metadata = np.load(meta_path, allow_pickle=True).tolist()
        return index, metadata

    # Empty new index
    index = faiss.IndexFlatL2(384)  # 384 for all-MiniLM-L6-v2
    metadata: List[Dict[str, Any]] = []
    return index, metadata

def persist_index(index: faiss.IndexFlatL2, metadata: List[Dict[str, Any]], storage_dir: str = "storage"):
    os.makedirs(storage_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(storage_dir, "faiss.index"))
    np.save(os.path.join(storage_dir, "meta.npy"), np.array(metadata, dtype=object), allow_pickle=True)

def add_texts_to_index(
    texts: List[str],
    source_name: str,
    embedder: SentenceTransformer,
    index: faiss.IndexFlatL2,
    metadata: List[Dict[str,Any]]
):
    if not texts:
      return
    vectors = embedder.encode(texts,convert_to_numpy=True,normalize_embeddings=False)
    index.add(vectors)
    for i,t in enumerate(texts):
      metadata.append({
          "source": source_name,
          "chunk_id": i,
          "text": t
      })

#Retriever
def retriever(query: str, embedder,index,metadata,k: int=3):
  if index.ntotal == 0:
    return []
  qvec = embedder.encode([query],convert_to_numpy=True)
  D, I = index.search(qvec, k=min(k, index.ntotal))
  results=[]
  for idx in I[0]:
    md = metadata(idx)
    results.append(md)
  return results

@st.cache_resource
def get_generator():
  return pipeline("text2text-generation",model="google/flan-t5-base") #flan-t5 is light and fine for grounded ans.

def synthesize_answer(user_query: str,contexts: List[Dict[str,Any]],history: List[Dict[str,str]],generator) -> str:
  conv = "\n".join([f"{m['role'].capitalize()}:{m['content']}" for m in history[-6:]])
  context_text = "\n\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(contexts)]) if contexts else "No context"
  prompt = f"""You are a helpful, concise support assistant. Answer the final user question using ONLY the provided context. If missing, say you don't know.
Coversation so far:
{conv}

context:
{context_text}

Final user question: {user_query}
Answer:"""
  out = generator(prompt,max_length=220,num_return_sequences=1)[0]["generated_text"]
  return out.strip()