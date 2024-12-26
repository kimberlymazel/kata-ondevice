import gradio as gr
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig

# --- Load Model ---
model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
quantization_config = QuantoConfig(weights="int8")
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=quantization_config, device_map="cpu"
)


# --- Helper Functions ---
def generate_response(user_input):
    """Generates a response using the LLaMA model."""
    start_time = time.time()
    messages = [
        {"role": "system", "content": "Tolong jawab singkat."},
        {"role": "user", "content": user_input},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Time tokenization
    tokenization_start = time.time()
    model_inputs = tokenizer([text], return_tensors="pt")
    tokenization_time = time.time() - tokenization_start

    # Time generation
    generation_start = time.time()
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    generation_time = time.time() - generation_start

    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    total_time = time.time() - start_time
    return (
        response,
        f"Tokenization Time: {tokenization_time:.2f}s",
        f"Generation Time: {generation_time:.2f}s",
        f"Total Time: {total_time:.2f}s",
    )


# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("### Isolated Response Generation App")
    input_text = gr.Textbox(label="Input Text", lines=2)
    response_text = gr.Textbox(label="Generated Response", lines=5, interactive=False)
    timing_info = gr.Textbox(label="Timing Info", lines=5, interactive=False)

    generate_button = gr.Button("Generate Response")

    generate_button.click(
        generate_response, inputs=[input_text], outputs=[response_text, timing_info]
    )

# Launch the Gradio app
demo.launch()
