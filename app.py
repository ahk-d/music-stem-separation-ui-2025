# app.py

import gradio as gr
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
import os
import tempfile
import numpy as np
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter
import warnings
import soundfile as sf
warnings.filterwarnings("ignore")

# --- Setup the models ---
print("Setting up models...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load HT-Demucs model
print("Loading HT-Demucs model...")
htdemucs_model = get_model(name="htdemucs")
htdemucs_model = htdemucs_model.to(device)
htdemucs_model.eval()
print("HT-Demucs model loaded successfully.")

# Load Spleeter model (5stems)
print("Loading Spleeter model...")
try:
    spleeter_separator = Separator('spleeter:5stems')
    spleeter_audio_adapter = AudioAdapter.default()
    print("Spleeter model loaded successfully.")
except Exception as e:
    print(f"Spleeter model loading failed: {e}")
    print("Trying with 2stems model as fallback...")
    try:
        spleeter_separator = Separator('spleeter:2stems')
        spleeter_audio_adapter = AudioAdapter.default()
        print("Spleeter 2stems model loaded successfully.")
    except Exception as e2:
        print(f"Spleeter 2stems model also failed: {e2}")
        spleeter_separator = None
        spleeter_audio_adapter = None

# --- HT-Demucs separation function ---
def separate_with_htdemucs(audio_path):
    """
    Separates an audio file using HT-Demucs into drums, bass, other, and vocals.
    Returns FILE PATHS.
    """
    if audio_path is None:
        return None, None, None, None, "Please upload an audio file."

    try:
        print(f"HT-Demucs: Loading audio from: {audio_path}")
        
        # Load audio with torchaudio
        wav, sr = torchaudio.load(audio_path)

        if wav.shape[0] == 1:
            print("Audio is mono, converting to stereo.")
            wav = wav.repeat(2, 1)

        wav = wav.to(device)

        print("HT-Demucs: Applying the separation model...")
        with torch.no_grad():
            sources = apply_model(htdemucs_model, wav[None], device=device, progress=True)[0]
        print("HT-Demucs: Separation complete.")

        # Save stems temporarily
        stem_names = ["drums", "bass", "other", "vocals"]
        output_dir = "htdemucs_stems"
        os.makedirs(output_dir, exist_ok=True)

        output_paths = []
        for i, name in enumerate(stem_names):
            out_path = os.path.join(output_dir, f"{name}.wav")
            torchaudio.save(out_path, sources[i].cpu(), sr)
            output_paths.append(out_path)
            print(f"✅ HT-Demucs saved {name} to {out_path}")

        return output_paths[0], output_paths[1], output_paths[2], output_paths[3], "✅ HT-Demucs separation successful!"

    except Exception as e:
        print(f"HT-Demucs Error: {e}")
        return None, None, None, None, f"❌ HT-Demucs Error: {str(e)}"

# --- Spleeter separation function ---
def separate_with_spleeter(audio_path):
    """
    Separates an audio file using Spleeter into vocals, drums, bass, other, and piano.
    Returns FILE PATHS.
    """
    if audio_path is None:
        return None, None, None, None, None, "Please upload an audio file."

    if spleeter_separator is None or spleeter_audio_adapter is None:
        return None, None, None, None, None, "❌ Spleeter model not loaded properly."

    try:
        print(f"Spleeter: Loading audio from: {audio_path}")
        
        # Load audio with Spleeter's audio adapter
        waveform, sample_rate = spleeter_audio_adapter.load(audio_path)
        print(f"Spleeter: Loaded audio with shape {waveform.shape}, sample rate {sample_rate}")
        
        print("Spleeter: Applying the separation model...")
        prediction = spleeter_separator.separate(waveform)
        print("Spleeter: Separation complete.")
        print(f"Spleeter: Prediction keys: {list(prediction.keys())}")
        
        # Debug: Check the shape of the first prediction
        first_key = list(prediction.keys())[0]
        print(f"Spleeter: Shape of {first_key}: {prediction[first_key].shape}")

        # Save stems temporarily
        output_dir = "spleeter_stems"
        os.makedirs(output_dir, exist_ok=True)

        output_paths = []
        
        # Handle different model types (5stems vs 2stems)
        if 'vocals' in prediction and 'drums' in prediction and 'bass' in prediction:
            # 5stems model
            stem_names = ["vocals", "drums", "bass", "other", "piano"]
            for name in stem_names:
                if name in prediction:
                    out_path = os.path.join(output_dir, f"{name}.wav")
                    stem_audio = prediction[name]
                    # Ensure audio is in the right format (2D array)
                    if stem_audio.ndim == 1:
                        stem_audio = stem_audio.reshape(1, -1)
                    spleeter_audio_adapter.save(out_path, stem_audio, 44100, 'wav', '16')
                    output_paths.append(out_path)
                    print(f"✅ Spleeter saved {name} to {out_path}")
                else:
                    output_paths.append(None)
        else:
            # 2stems model (vocals + accompaniment)
            stem_names = ["vocals", "accompaniment"]
            for name in stem_names:
                if name in prediction:
                    out_path = os.path.join(output_dir, f"{name}.wav")
                    stem_audio = prediction[name]
                    # Ensure audio is in the right format (2D array)
                    if stem_audio.ndim == 1:
                        stem_audio = stem_audio.reshape(1, -1)
                    spleeter_audio_adapter.save(out_path, stem_audio, 44100, 'wav', '16')
                    output_paths.append(out_path)
                    print(f"✅ Spleeter saved {name} to {out_path}")
                else:
                    output_paths.append(None)
            
            # Fill remaining slots with None for 2stems model
            while len(output_paths) < 5:
                output_paths.append(None)

        return output_paths[0], output_paths[1], output_paths[2], output_paths[3], output_paths[4], "✅ Spleeter separation successful!"

    except Exception as e:
        print(f"Spleeter Error: {e}")
        return None, None, None, None, None, f"❌ Spleeter Error: {str(e)}"

# --- Combined separation function ---
def separate_selected_models(audio_path, run_htdemucs, run_spleeter):
    """
    Separates an audio file using selected models (HT-Demucs, Spleeter, or both).
    Returns stems from selected models.
    """
    if audio_path is None:
        return [None] * 13, "Please upload an audio file."

    if not run_htdemucs and not run_spleeter:
        return [None] * 13, "❌ Please select at least one model to run."

    try:
        htdemucs_results = [None] * 5  # 4 stems + 1 status
        spleeter_results = [None] * 6  # 5 stems + 1 status
        status_messages = []
        
        # Run HT-Demucs if selected
        if run_htdemucs:
            print("Running HT-Demucs...")
            htdemucs_results = separate_with_htdemucs(audio_path)
            status_messages.append(htdemucs_results[-1])
        
        # Run Spleeter if selected
        if run_spleeter:
            print("Running Spleeter...")
            spleeter_results = separate_with_spleeter(audio_path)
            status_messages.append(spleeter_results[-1])
        
        # Combine results: HT-Demucs (4 stems) + Spleeter (5 stems) + status messages
        all_results = list(htdemucs_results[:-1]) + list(spleeter_results[:-1]) + status_messages
        
        # Create combined status message
        models_used = []
        if run_htdemucs:
            models_used.append("HT-Demucs")
        if run_spleeter:
            models_used.append("Spleeter")
        
        combined_status = f"🎵 {' + '.join(models_used)} completed!\n\n" + "\n".join(status_messages)
        
        return all_results + [combined_status]

    except Exception as e:
        print(f"Combined Error: {e}")
        return [None] * 13, f"❌ Error: {str(e)}"

# --- Gradio UI ---
print("Creating Gradio interface...")
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎵 Music Stem Separator - HT-Demucs & Spleeter Comparison
    
    Upload your music and get stems from both **HT-Demucs** and **Spleeter** models!
    
    **HT-Demucs** provides: Drums, Bass, Other, Vocals  
    **Spleeter** provides: Vocals, Drums, Bass, Other, **Piano** 🎹
    
    Compare the quality and choose the best stems for your needs!
    """)

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(type="filepath", label="🎵 Upload Your Song")
            
            # Model selection toggles
            gr.Markdown("### 🎛️ Select Models to Run")
            with gr.Row():
                htdemucs_toggle = gr.Checkbox(label="🎯 HT-Demucs", value=True, info="Drums, Bass, Other, Vocals")
                spleeter_toggle = gr.Checkbox(label="🎵 Spleeter", value=True, info="Vocals, Drums, Bass, Other, Piano")
            
            separate_button = gr.Button("🚀 Separate Music", variant="primary", size="lg")
            status_output = gr.Textbox(label="📊 Status", interactive=False, lines=4)

    gr.Markdown("---")
    
    with gr.Row():
        # HT-Demucs Results
        with gr.Column():
            gr.Markdown("### 🎯 HT-Demucs Results")
            with gr.Row():
                htdemucs_drums = gr.Audio(label="🥁 Drums", type="filepath")
                htdemucs_bass = gr.Audio(label="🎸 Bass", type="filepath")
            with gr.Row():
                htdemucs_other = gr.Audio(label="🎼 Other", type="filepath")
                htdemucs_vocals = gr.Audio(label="🎤 Vocals", type="filepath")
        
        # Spleeter Results
        with gr.Column():
            gr.Markdown("### 🎵 Spleeter Results")
            with gr.Row():
                spleeter_vocals = gr.Audio(label="🎤 Vocals", type="filepath")
                spleeter_drums = gr.Audio(label="🥁 Drums", type="filepath")
            with gr.Row():
                spleeter_bass = gr.Audio(label="🎸 Bass", type="filepath")
                spleeter_other = gr.Audio(label="🎼 Other", type="filepath")
            with gr.Row():
                spleeter_piano = gr.Audio(label="🎹 Piano", type="filepath")
            
            gr.Markdown("*Note: If only 2 stems are available, only Vocals and Accompaniment will be shown*")

    gr.Markdown("---")
    
    with gr.Row():
        gr.Markdown("""
        ### 📋 Model Comparison
        
        | Feature | HT-Demucs | Spleeter |
        |---------|-----------|----------|
        | **Vocals** | ✅ High Quality | ✅ High Quality |
        | **Drums** | ✅ High Quality | ✅ High Quality |
        | **Bass** | ✅ High Quality | ✅ High Quality |
        | **Other** | ✅ High Quality | ✅ High Quality |
        | **Piano** | ❌ Not Available | ✅ **Available** |
        | **Speed** | ⚡ Fast | ⚡ Fast |
        | **Quality** | 🏆 Excellent | 🏆 Excellent |
        
        **💡 Tip:** Use Spleeter when you need piano separation, HT-Demucs for other instruments!
        """)

    # Connect the button to the combined function
    separate_button.click(
        fn=separate_selected_models,
        inputs=[audio_input, htdemucs_toggle, spleeter_toggle],
        outputs=[
            htdemucs_drums, htdemucs_bass, htdemucs_other, htdemucs_vocals,  # HT-Demucs outputs
            spleeter_vocals, spleeter_drums, spleeter_bass, spleeter_other, spleeter_piano,  # Spleeter outputs
            status_output  # Status output
        ]
    )

    gr.Markdown("""
    ---
    <p style='text-align: center; font-size: small;'>
    🚀 Powered by <strong>HT-Demucs</strong> & <strong>Spleeter</strong> | 
    🎵 Compare and choose your best stems!
    </p>
    """)

if __name__ == "__main__":
    demo.launch(share=True)