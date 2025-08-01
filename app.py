# app.py

# 2. Import Libraries
import gradio as gr
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
import os
import time

# 3. Setup the Model
# This section sets up the device (GPU if available) and loads the pre-trained HT Demucs model.
print("Setting up the model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the pre-trained HTDemucs model
# To make this work on Hugging Face, we'll download the model weights to a cache folder.
# The `get_model` function handles this automatically.
model = get_model(name="htdemucs")
model = model.to(device)
model.eval()
print("Model loaded successfully.")

# 4. Define the Separation Function
def separate_stems(audio_path):
    """
    This function takes an audio file path, separates it into stems,
    and returns the paths to the separated audio files.
    """
    if audio_path is None:
        return None, None, None, None, "Please upload an audio file."

    try:
        print(f"Loading audio from: {audio_path}")
        # Load the audio file
        wav, sr = torchaudio.load(audio_path)

        # Ensure the audio is stereo
        if wav.shape[0] == 1:
            print("Audio is mono, converting to stereo.")
            wav = wav.repeat(2, 1)

        # Move tensor to the correct device
        wav = wav.to(device)

        # Apply the separation model
        print("Applying the separation model...")
        with torch.no_grad():
            # The apply_model function expects a batch, so we add a dimension
            sources = apply_model(model, wav[None], device=device, progress=True)[0]
        print("Separation complete.")

        # Define stem names
        stem_names = ["drums", "bass", "other", "vocals"]

        # Create a directory to save the output files
        # It's good practice to use a temporary directory for each session
        # or a unique folder to avoid conflicts in a multi-user environment
        output_dir = "separated_stems"
        os.makedirs(output_dir, exist_ok=True)

        # Save each stem and collect their paths
        output_paths = []
        for i, name in enumerate(stem_names):
            out_path = os.path.join(output_dir, f"{name}.wav")
            torchaudio.save(out_path, sources[i].cpu(), sr)
            output_paths.append(out_path)
            print(f"Saved {name} stem to {out_path}")

        # Return the paths to the separated audio files
        return output_paths[0], output_paths[1], output_paths[2], output_paths[3], "Separation successful!"

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None, f"An error occurred: {str(e)}"

# 5. Create the Gradio Interface
print("Creating Gradio interface...")
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎵 Music Stem Separator with HT Demucs
        Upload your song (in .wav or .mp3 format) and the model will separate it into four stems: **Drums**, **Bass**, **Other**, and **Vocals**.
        """
    )

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="Upload Your Song")
            separate_button = gr.Button("Separate Music", variant="primary")
            status_output = gr.Textbox(label="Status", interactive=False)

        with gr.Column():
            gr.Markdown("### Separated Stems")
            drums_output = gr.Audio(label="Drums", type="filepath")
            bass_output = gr.Audio(label="Bass", type="filepath")
            other_output = gr.Audio(label="Other", type="filepath")
            vocals_output = gr.Audio(label="Vocals", type="filepath")

    separate_button.click(
        fn=separate_stems,
        inputs=audio_input,
        outputs=[drums_output, bass_output, other_output, vocals_output, status_output]
    )

    gr.Markdown(
        """
        ---
        <p style='text-align: center; font-size: small;'>
        Powered by <a href='https://github.com/facebookresearch/demucs' target='_blank'>HT Demucs</a>.
        </p>
        """
    )

# 6. Launch the Gradio App
# The launch command should be at the end of the script
demo.launch()