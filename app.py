
import os
import torch
import gradio as gr
from ddgs import DDGS
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "v2_model")

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map="cpu"
)
model = PeftModel.from_pretrained(base, ADAPTER_PATH)
model.eval()

def ask_ai(message, history=None):
    messages = [
        {"role": "system", "content":
         "You are NexaLLM, a helpful AI assistant. "
         "Answer clearly and honestly."}
    ]
    for user_msg, bot_msg in (history or []):
        messages.extend([
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": bot_msg}
        ])
    messages.append({"role": "user", "content": message})

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=200, do_sample=False
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

def search_web(query):
    try:
        results = list(DDGS().text(query, max_results=5))
        if not results:
            return "No search results found."
        summary = ask_ai(
            "Answer using these search results. Include source URLs.\n"
            + "\n".join(
                f"{r.get('title', '')}: {r.get('body', '')} "
                f"URL: {r.get('href', '')}" for r in results
            )
        )
        sources = "\n\n".join(
            f"- [{r.get('title', 'Source')}]({r.get('href', '')})"
            for r in results
        )
        return summary + "\n\n### Sources\n" + sources
    except Exception as e:
        return f"Web search error: {e}"

with gr.Blocks(title="NexaLLM") as demo:
    gr.Markdown("# NexaLLM\n### An LLM-Powered Intelligent Web Assistant")
    with gr.Tab("AI Chatbot"):
        gr.ChatInterface(fn=ask_ai)
    with gr.Tab("Web Search"):
        query = gr.Textbox(label="Search the web")
        result = gr.Markdown()
        gr.Button("Search").click(search_web, query, result)
        query.submit(search_web, query, result)

demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "10000")))
