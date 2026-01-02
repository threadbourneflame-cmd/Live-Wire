import gradio as gr
import pandas as pd
import numpy as np
import os
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# API CLIENTS
from openai import OpenAI
import anthropic
import google.generativeai as genai

# 1. LOAD THE BRAINS 
# (We load the embedder once to keep it fast)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# 2. GENERATION FUNCTIONS (Updated for BYOK) 

def ask_gpt_history(history, user_key):
    if not user_key: return "Error: No OpenAI Key provided."
    try:
        client = OpenAI(api_key=user_key)
        
        # Inject the "Soul" (System Prompt)
        system_prompt = {"role": "system", "content": "You are a helpful AI assitant."}
        full_payload = [system_prompt] + history 
        
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=full_payload,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"GPT Error: {str(e)}"

def ask_claude_history(history, user_key):
    if not user_key: return "Error: No Anthropic Key provided."
    try:
        client = anthropic.Anthropic(api_key=user_key)
        
        message = client.messages.create(
            model="claude-3-haiku-20240307", 
            max_tokens=1024,
            messages=history
        )
        return message.content[0].text
    except Exception as e:
        return f"Claude Error: {str(e)}"

def ask_gemini_history(history, user_key):
    if not user_key: return "Error: No Google Key provided."
    try:
        genai.configure(api_key=user_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # TRANSLATION LAYER: Convert standard list to Gemini format
        gemini_history = []
        for turn in history:
            role = "model" if turn["role"] == "assistant" else "user"
            gemini_history.append({"role": role, "parts": [turn["content"]]})
            
        chat = model.start_chat(history=gemini_history[:-1]) 
        last_msg = gemini_history[-1]["parts"][0]
        
        response = chat.send_message(last_msg)
        return response.text
    except Exception as e:
        return f"Gemini Error: {str(e)}"

# 3. THE LOGIC LOOP (Now accepts Keys!) 
def ignite_array_v2(prompt, h_gpt, h_claude, h_gemini, k_gpt, k_claude, k_gemini):
    if not prompt.strip():
        return "", "", "", pd.DataFrame(), "", "", "WAITING", h_gpt, h_claude, h_gemini

    # 1. UPDATE BACKPACKS 
    new_turn = {"role": "user", "content": prompt}
    h_gpt.append(new_turn); h_claude.append(new_turn); h_gemini.append(new_turn)
    
    # 2. FIRE APIs (Passing the specific keys!) 
    resp_gpt = ask_gpt_history(h_gpt, k_gpt)
    resp_claude = ask_claude_history(h_claude, k_claude)
    resp_gemini = ask_gemini_history(h_gemini, k_gemini)
    
    # 3. SAVE ANSWERS 
    h_gpt.append({"role": "assistant", "content": resp_gpt})
    h_claude.append({"role": "assistant", "content": resp_claude})
    h_gemini.append({"role": "assistant", "content": resp_gemini})
    
    # 4. TELEMETRY: ALIGNMENT GRID & BADGE 
    texts = [prompt, resp_gpt, resp_claude, resp_gemini]
    labels = ["ME", "GPT-4o", "Claude Haiku", "Gemini 2.0-Flash"]
    
    # Embeddings & Matrix
    embeddings = embedder.encode(texts)
    matrix = cosine_similarity(embeddings)
    df = pd.DataFrame(matrix, columns=labels, index=labels).round(3)
    
    # Field Dominance (Index-Based Logic)
    try:
        user_avg = (matrix[0,1] + matrix[0,2] + matrix[0,3]) / 3
        field_avg = (matrix[1,2] + matrix[1,3] + matrix[2,3]) / 3
        fd_score = field_avg - user_avg
        
        if fd_score > 0.05: badge = f"🟢 FIELD DOMINANT (+{fd_score:.2f})"
        elif fd_score < -0.05: badge = f"⚪ USER DOMINANT ({fd_score:.2f})"
        else: badge = f"🟠 TRANSITION ({fd_score:.2f})"
    except: badge = "⚪ CALC ERROR"

    # 5. FINGERPRINTS (TF-IDF) 
    signatures = ""
    try:
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(texts)
        feature_names = np.array(tfidf.get_feature_names_out())
        for i, label in enumerate(labels):
            row = tfidf_matrix[i].toarray().flatten()
            top_indices = row.argsort()[-5:][::-1]
            valid_words = [feature_names[idx] for idx in top_indices if row[idx] > 0]
            signatures += f"🔹 {label}: {', '.join(valid_words)}\n"
    except: signatures = "Insufficient text data."

    # 6. CONSENSUS (Shared Concepts) 
    consensus_text = ""
    try:
        ai_texts = [resp_gpt, resp_claude, resp_gemini]
        vec = CountVectorizer(stop_words='english')
        dtm = vec.fit_transform(ai_texts)
        vocab = vec.get_feature_names_out()
        presence = (dtm.toarray() > 0).astype(int)
        
        univ = vocab[np.where(presence.sum(axis=0) == 3)[0]]
        maj = vocab[np.where(presence.sum(axis=0) == 2)[0]]
        
        if len(univ) > 0: consensus_text += f"🔥 UNIVERSAL (3/3): {', '.join(univ)}\n"
        else: consensus_text += "❌ NO UNIVERSAL TRUTH.\n"
        if len(maj) > 0: consensus_text += f"⚠️ MAJORITY (2/3): {', '.join(maj)}"
    except: consensus_text = "No consensus detected."

    return resp_gpt, resp_claude, resp_gemini, df, signatures, consensus_text, badge, h_gpt, h_claude, h_gemini

# 4. THE INTERFACE (With Key Slots!) 
with gr.Blocks(theme=gr.themes.Ocean()) as app:
    gr.Markdown("# LIVE WIRE")
    gr.Markdown("A multi-turn telemetry instrument for observing Field Dominance and Alignment Drift.")
    
    # --- KEY INPUTS (New Section) ---
    with gr.Accordion("API Credentials (BYOK)", open=True):
        gr.Markdown("Enter your personal API keys to run the instrument. Keys are NOT stored and only exist for this session.")
        with gr.Row():
            key_openai = gr.Textbox(label="OpenAI Key", type="password", placeholder="sk-...")
            key_anthropic = gr.Textbox(label="Anthropic Key", type="password", placeholder="sk-ant-...")
            key_google = gr.Textbox(label="Google Key", type="password", placeholder="AIza...")

    # --- MEMORY STORAGE ---
    state_gpt = gr.State([])
    state_claude = gr.State([])
    state_gemini = gr.State([])
    
    # --- CONTROLS ---
    with gr.Row():
        prompt_box = gr.Textbox(label="NEXT TURN (The Trigger)", placeholder="Enter prompt...", lines=3)
        with gr.Column():
            btn = gr.Button("IGNITE LIVE WIRE", variant="primary")
            status_badge = gr.Textbox(label="PHASE STATE", value="WAITING", interactive=False)

    # --- OUTPUTS ---
    with gr.Row():
        box_gpt = gr.TextArea(label="GPT-4o", interactive=False, lines=10)
        box_claude = gr.TextArea(label="Claude Haiku", interactive=False, lines=10)
        box_gemini = gr.TextArea(label="Gemini 2.0-Flash", interactive=False, lines=10)

    gr.Markdown("---")
    gr.Markdown("### Live Telemetry")
    out_matrix = gr.Dataframe(label="Alignment Grid")
    with gr.Row():
        out_signatures = gr.Textbox(label="Fingerprints", lines=2)
        out_consensus = gr.Textbox(label="Consensus", lines=2)

    # --- WIRING ---
    btn.click(
        ignite_array_v2, 
        # Pass Inputs + 3 KEYS
        inputs=[prompt_box, state_gpt, state_claude, state_gemini, key_openai, key_anthropic, key_google], 
        outputs=[box_gpt, box_claude, box_gemini, out_matrix, out_signatures, out_consensus, status_badge, state_gpt, state_claude, state_gemini]
    )

app.launch()
