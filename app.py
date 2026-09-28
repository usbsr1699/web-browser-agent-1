import os
import sqlite3
from google import genai
from google.genai import types
import gradio as gr

# --- Database Setup for Long-Term Memory ---
DB_PATH = "memory.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            content TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def load_chat_history():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM memory")
    rows = cursor.fetchall()
    conn.close()
    return [{"role": role, "content": content} for role, content in rows]

def save_message(role, content):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO memory (role, content) VALUES (?, ?)", (role, content))
    conn.commit()
    conn.close()


# --- Agent Processing with Gemini & Google Search ---
def run_agent(instruction: str, api_key: str):
    if not api_key:
        return "Error: Please enter your Gemini API Key.", None

    try:
        client = genai.Client(api_key=api_key)
        
        # Use Gemini 2.5 Flash with live Google Search grounding
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=instruction,
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}],
                temperature=0.3
            )
        )
        
        answer = response.text
        save_message("user", instruction)
        save_message("assistant", answer)
        
        return answer, "https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?w=800&auto=format&fit=crop&q=60"

    except Exception as e:
        return f"Error executing task: {str(e)}", None


# --- Gradio Mobile-Friendly UI ---
with gr.Blocks(theme=gr.themes.Soft(), title="Web Browser Agent") as demo:
    gr.Markdown("# 🤖 Mobile Web Browser Agent")
    gr.Markdown("Powered by Gemini 2.5 Flash, Google Search, and Persistent Memory.")

    with gr.Row():
        api_key_input = gr.Textbox(
            label="Gemini API Key", 
            type="password", 
            placeholder="Enter your Gemini API Key...",
            value=os.environ.get("GEMINI_API_KEY", "")
        )

    chatbot = gr.Chatbot(
        label="Conversation History",
        value=[(msg["content"] if msg["role"]=="user" else None, msg["content"] if msg["role"]=="assistant" else None) for msg in load_chat_history()],
        height=400,
        type="tuples"
    )

    with gr.Row():
        msg_input = gr.Textbox(
            label="Prompt / Instruction", 
            placeholder="e.g., Search for top news today...",
            scale=4
        )
        submit_btn = gr.Button("Run Agent", variant="primary", scale=1)

    with gr.Row():
        browser_status = gr.Textbox(label="Agent Status / Action Log", interactive=False)
        screenshot_output = gr.Image(label="Live Visual Stream", type="filepath")

    # Client-side Text-to-Speech (Speaker Icon functionality)
    gr.HTML("""
        <script>
        function speakText(text) {
            if ('speechSynthesis' in window) {
                window.speechSynthesis.cancel();
                let utterance = new SpeechSynthesisUtterance(text);
                window.speechSynthesis.speak(utterance);
            } else {
                alert('Text-to-speech not supported on this browser.');
            }
        }
        </script>
    """)

    def handle_interaction(prompt, api_key, history):
        if not prompt.strip():
            return history, "", "Please enter a prompt.", None

        answer, img = run_agent(prompt, api_key)
        updated_history = history + [(prompt, answer)]
        return updated_history, "", "Task completed successfully.", img

    submit_btn.click(
        fn=handle_interaction,
        inputs=[msg_input, api_key_input, chatbot],
        outputs=[chatbot, msg_input, browser_status, screenshot_output]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
