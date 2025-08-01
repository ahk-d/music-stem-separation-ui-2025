# app.py

import gradio as gr
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
import os
import base64

# --- Setup the model ---
print("Setting up the model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = get_model(name="htdemucs")
model = model.to(device)
model.eval()
print("Model loaded successfully.")

# --- Helper function to convert WAV to base64 data URI ---
def file_to_data_uri(path):
    with open(path, "rb") as f:
        data = f.read()
    return f"data:audio/wav;base64,{base64.b64encode(data).decode()}"

# --- Separation function ---
def separate_stems(audio_path):
    """
    Separates an audio file into drums, bass, other, and vocals.
    Returns base64-encoded audio URIs for frontend playback.
    """
    if audio_path is None:
        return None, None, None, None, "Please upload an audio file."

    try:
        print(f"Loading audio from: {audio_path}")
        wav, sr = torchaudio.load(audio_path)

        if wav.shape[0] == 1:
            print("Audio is mono, converting to stereo.")
            wav = wav.repeat(2, 1)

        wav = wav.to(device)

        print("Applying the separation model...")
        with torch.no_grad():
            sources = apply_model(model, wav[None], device=device, progress=True)[0]
        print("Separation complete.")

        # Save stems temporarily & encode to base64 URIs
        stem_names = ["drums", "bass", "other", "vocals"]
        output_dir = "separated_stems"
        os.makedirs(output_dir, exist_ok=True)

        output_uris = []
        for i, name in enumerate(stem_names):
            out_path = os.path.join(output_dir, f"{name}.wav")
            torchaudio.save(out_path, sources[i].cpu(), sr)
            output_uris.append(file_to_data_uri(out_path))
            print(f"Encoded {name} to base64 URI")

        return output_uris[0], output_uris[1], output_uris[2], output_uris[3], "✅ Separation successful!"

    except Exception as e:
        print(f"Error: {e}")
        return None, None, None, None, f"❌ Error: {str(e)}"

# --- Gradio UI ---
print("Creating Gradio interface...")
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎵 Music Stem Separator with HT Demucs")

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="Upload Your Song")
            separate_button = gr.Button("Separate Music", variant="primary")
            status_output = gr.Textbox(label="Status", interactive=False)

        with gr.Column():
            gr.Markdown("### 🎧 Separated Stems")
            drums_output = gr.Audio(label="Drums", type="filepath")
            bass_output = gr.Audio(label="Bass", type="filepath")
            other_output = gr.Audio(label="Other", type="filepath")
            vocals_output = gr.Audio(label="Vocals", type="filepath")

    separate_button.click(
        fn=separate_stems,
        inputs=audio_input,
        outputs=[drums_output, bass_output, other_output, vocals_output, status_output]
    )

    gr.Markdown("---\n<p style='text-align: center; font-size: small;'>Powered by HT Demucs</p>")

# ✅ Enable API for Next.js
demo.launch(share=True)
