import os
import streamlit as st
from typing import List
import openai

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

import numpy as np
import pinecone

st.set_page_config(page_title="Chatbot QA")


def _get_secret(key: str, default=None):
    val = os.getenv(key)
    if not val:
        val = st.secrets.get(key, None) if hasattr(st, 'secrets') else None
    return val if val is not None else default


OPENAI_API_KEY = _get_secret('OPENAI_API_KEY')
PINECONE_API_KEY = _get_secret('PINECONE_API_KEY')
PINECONE_ENV = _get_secret('PINECONE_ENV')
INDEX_NAME = _get_secret('PINECONE_INDEX', 'chatbot-index')
EMBEDDINGS_MODEL = _get_secret('EMBEDDINGS_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')

if not OPENAI_API_KEY:
    st.error('Falta OPENAI_API_KEY en el entorno o .streamlit/secrets.toml')

openai.api_key = OPENAI_API_KEY


@st.cache_resource
def get_embedding_model():
    if SentenceTransformer is None:
        raise RuntimeError('sentence_transformers no disponible; instálalo en el entorno.')
    return SentenceTransformer(EMBEDDINGS_MODEL)


@st.cache_resource
def get_pinecone_index():
    if not PINECONE_API_KEY or not PINECONE_ENV:
        raise RuntimeError('Faltan PINECONE_API_KEY/PINECONE_ENV')
    # New pinecone client (v7+) exposes a Pinecone class. Create a client instance
    try:
        client = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    except Exception as e:
        raise RuntimeError(f'No se pudo inicializar el cliente de Pinecone: {e}')

    # list_indexes may return different shapes depending on version; be permissive
    try:
        idxs = client.list_indexes()
        if hasattr(idxs, 'names'):
            available = idxs.names()
        elif isinstance(idxs, list):
            available = idxs
        elif hasattr(idxs, 'results'):
            # results might be list of objects with .name
            available = [getattr(r, 'name', r) for r in idxs.results]
        else:
            # fallback: try to coerce to list
            available = list(idxs)
    except Exception as e:
        raise RuntimeError(f'No se pudo listar índices en Pinecone: {e}')

    if INDEX_NAME not in available:
        raise RuntimeError(f'Índice {INDEX_NAME} no existe. Índices disponibles: {available}. Ejecuta ingest.py primero para crear y poblar el índice.')

    # Obtain an Index object bound to this client. Different client versions expose
    # the Index factory in different places; try the client first, then the top-level
    try:
        if hasattr(client, 'Index'):
            idx = client.Index(INDEX_NAME)
        else:
            # fallback to top-level Index (older shape)
            idx = pinecone.Index(INDEX_NAME)
    except Exception as e:
        raise RuntimeError(f'No se pudo obtener el objeto Index de Pinecone: {e}')

    return idx


def embed_text(text: str):
    model = get_embedding_model()
    v = model.encode([text])[0]
    # normalize to numpy array of floats
    arr = np.array(v, dtype=float)
    # check for NaN / Inf
    if not np.isfinite(arr).all():
        raise RuntimeError('El embedding contiene NaN o Inf; revisa el modelo de embeddings.')
    # convert to native Python floats (JSON-serializable)
    return [float(x) for x in arr.tolist()]


def query_index(query: str, top_k: int = 5):
    idx = get_pinecone_index()
    qv = embed_text(query)
    # Ensure qv is a flat list of native floats
    if not isinstance(qv, list):
        qv = list(qv)
    qv = [float(x) for x in qv]

    # Try primary query signature; fall back to alternate argument names if needed
    last_err = None
    try:
        res = idx.query(queries=[qv], top_k=top_k, include_metadata=True)
    except Exception as e:
        last_err = e
        try:
            # older/newer clients may accept vector= or top_k named differently
            res = idx.query(vector=qv, top_k=top_k, include_metadata=True)
        except Exception as e2:
            # raise the original error but include fallback info
            raise RuntimeError(f'Error al consultar Pinecone. Intentos: primary -> {last_err}; fallback -> {e2}')
    # handle different pinecone client shapes
    if 'results' in res:
        matches = res['results'][0].get('matches', [])
    else:
        matches = res.get('matches', [])
    return matches


st.title('Chatbot Preguntas y Respuestas IA')

if 'messages' not in st.session_state:
    st.session_state['messages'] = []



with st.chat_message('assistant'):
    st.write('Hola, ¿en qué puedo ayudarte hoy?')

user_input = st.chat_input('Escribe tu pregunta aquí...')
if user_input:
    with st.chat_message('user'):
        st.write(user_input)

    # Recuperar contexto
    try:
        matches = query_index(user_input, top_k=8)
        context_texts = []
        for m in matches:
            meta = m.get('metadata', {})
            src = meta.get('source', '')
            txt = meta.get('text', '')
            # truncate each chunk to avoid sending huge payloads; keep more detail if needed
            max_chunk_chars = 1500
            txt_short = txt if len(txt) <= max_chunk_chars else txt[:max_chunk_chars] + '...'
            score = m.get('score') or m.get('distance') or m.get('score', None)
            if score is None and 'match' in m:
                score = m['match'].get('score') if isinstance(m.get('match'), dict) else None
            score_str = f" (score={score:.4f})" if isinstance(score, (float,)) else ''
            context_texts.append(f"Fuente: {src}{score_str}\n{txt_short}")
        context = '\n---\n'.join(context_texts)
    except Exception as e:
        context = ''
        st.warning(f'No se pudo recuperar contexto de Pinecone: {e}')

    
    # Construir prompt
    # Stronger prompt: force use of context and avoid hallucination
    system = (
        "Eres un asistente experto que RESPONDE SÓLO usando el contexto proporcionado. "
        "Si la respuesta no está en el contexto, di claramente que no tienes suficiente información. "
        "Cita la(s) fuente(s) cuando correspondan.")

    # Limit the total context size to ~3000 chars to avoid token overflow
    max_total_context_chars = 3000
    if context and len(context) > max_total_context_chars:
        context = context[:max_total_context_chars] + '\n...[context truncated]'

    user_msg = f"Pregunta: {user_input}\n\nContexto:\n{context}\n\nResponde de forma concisa, usando SOLO el contexto anterior."

    try:
        resp = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user_msg}
            ],
            temperature=0.2,
        )
        answer = resp['choices'][0]['message']['content']
    except Exception as e:
        answer = f'Error al llamar a OpenAI: {e}'

    st.session_state['messages'].append((user_input, answer))

    with st.chat_message('assistant'):
        st.write(answer)
