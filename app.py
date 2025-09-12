# app.py

import gradio as gr
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
import os
import tempfile
import numpy as np
import warnings
import soundfile as sf
import librosa
import time
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

# Setup Spleeter with Python API approach
print("Setting up Spleeter...")
spleeter_separator = None
spleeter_audio_adapter = None
spleeter_available = False

def patch_spleeter_redirects():
    """Patch Spleeter to handle GitHub redirects properly"""
    try:
        import httpx
        from spleeter.model.provider.github import GithubModelProvider
        
        # Store the original download method
        original_download = GithubModelProvider.download
        
        def patched_download(self, name, model_directory):
            """Patched download method that handles redirects"""
            import os
            import tarfile
            import tempfile
            from urllib.parse import urlparse
            
            print(f"Downloading {name} model with redirect handling...")
            
            # Model URLs - only 5stems
            model_urls = {
                '5stems': 'https://github.com/deezer/spleeter/releases/download/v1.4.0/5stems.tar.gz'
            }
            
            if name not in model_urls:
                return original_download(self, name, model_directory)
            
            url = model_urls[name]
            
            try:
                # Create a session that follows redirects
                with httpx.Client(follow_redirects=True, timeout=300) as client:
                    print(f"Downloading from: {url}")
                    response = client.get(url)
                    response.raise_for_status()
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.tar.gz') as tmp_file:
                        tmp_file.write(response.content)
                        tmp_file_path = tmp_file.name
                    
                    print(f"Downloaded {len(response.content)} bytes")
                    
                    # Extract the model
                    os.makedirs(model_directory, exist_ok=True)
                    with tarfile.open(tmp_file_path, 'r:gz') as tar:
                        tar.extractall(model_directory)
                    
                    # Clean up
                    os.unlink(tmp_file_path)
                    print(f"✅ Successfully downloaded and extracted {name} model")
                    
            except Exception as e:
                print(f"❌ Failed to download {name} model: {e}")
                # Fallback to original method
                return original_download(self, name, model_directory)
        
        # Apply the patch
        GithubModelProvider.download = patched_download
        print("✅ Patched Spleeter to handle GitHub redirects")
        return True
        
    except Exception as e:
        print(f"⚠️ Could not patch Spleeter redirects: {e}")
        return False

def setup_spleeter_with_retry():
    """Setup Spleeter 5stems model only"""
    global spleeter_separator, spleeter_audio_adapter, spleeter_available
    
    try:
        from spleeter.separator import Separator
        from spleeter.audio.adapter import AudioAdapter
        import os
        
        # Patch Spleeter to handle redirects
        patch_spleeter_redirects()
        
        # Set environment variables to help with model download
        os.environ['SPLEETER_MODEL_PATH'] = '/tmp/spleeter_models'
        
        # Create the 5stems separator
        print("Creating Spleeter 5stems separator...")
        spleeter_separator = Separator('spleeter:5stems')
        spleeter_audio_adapter = AudioAdapter.default()
        spleeter_available = True
        print("✅ Spleeter 5stems model loaded successfully!")
        return True
                
    except Exception as e:
        print(f"❌ Failed to load Spleeter 5stems: {e}")
        spleeter_separator = None
        spleeter_audio_adapter = None
        spleeter_available = False
        return False

# Try to setup Spleeter
setup_spleeter_with_retry()

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

        # Save stems with timestamp to ensure uniqueness
        timestamp = int(time.time() * 1000)  # millisecond timestamp
        output_dir = f"htdemucs_stems_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        stem_names = ["drums", "bass", "other", "vocals"]

        output_paths = []
        for i, name in enumerate(stem_names):
            out_path = os.path.join(output_dir, f"{name}_{timestamp}.wav")
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
    Uses Python API approach from stem_separation_spleeter.py
    Returns FILE PATHS.
    """
    if audio_path is None:
        return None, None, None, None, None, "Please upload an audio file."

    if not spleeter_available or spleeter_separator is None or spleeter_audio_adapter is None:
        return None, None, None, None, None, "❌ Spleeter not available. Please install Spleeter."

    try:
        print(f"Spleeter: Processing audio from: {audio_path}")
        
        # Create output directory with timestamp
        timestamp = int(time.time() * 1000)
        output_dir = f"spleeter_stems_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Load audio using Spleeter's audio adapter (from stem_separation_spleeter.py)
        print("Spleeter: Loading audio...")
        waveform, sample_rate = spleeter_audio_adapter.load(audio_path, sample_rate=44100)
        print(f"Spleeter: Loaded audio - shape: {waveform.shape}, sr: {sample_rate}")
        
        # Perform the separation (from stem_separation_spleeter.py)
        print("Spleeter: Separating audio sources...")
        prediction = spleeter_separator.separate(waveform)
        print("Spleeter: Separation complete.")
        print(f"Spleeter: Prediction keys: {list(prediction.keys())}")
        
        # Save stems with timestamp
        output_paths = []
        stem_names = ["vocals", "drums", "bass", "other", "piano"]
        
        for stem_name in stem_names:
            if stem_name in prediction:
                out_path = os.path.join(output_dir, f"{stem_name}_{timestamp}.wav")
                stem_audio = prediction[stem_name]
                
                print(f"Spleeter: {stem_name} audio shape: {stem_audio.shape}, dtype: {stem_audio.dtype}")
                
                # Save using soundfile for better compatibility
                sf.write(out_path, stem_audio, sample_rate)
                output_paths.append(out_path)
                print(f"✅ Spleeter saved {stem_name} to {out_path}")
            else:
                print(f"⚠️ Warning: {stem_name} not found in prediction")
                output_paths.append(None)
        
        # Ensure we have 5 outputs
        while len(output_paths) < 5:
            output_paths.append(None)

        return output_paths[0], output_paths[1], output_paths[2], output_paths[3], output_paths[4], "✅ Spleeter separation successful!"

    except Exception as e:
        print(f"Spleeter Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, f"❌ Spleeter Error: {str(e)}"

# --- Combined separation function ---
def separate_selected_models(audio_path, run_htdemucs, run_spleeter):
    """
    Separates an audio file using selected models (HT-Demucs, Spleeter, or both).
    Returns stems from selected models.
    """
    if audio_path is None:
        return [None] * 11, "Please upload an audio file."

    if not run_htdemucs and not run_spleeter:
        return [None] * 11, "❌ Please select at least one model to run."

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
        
        # Combine results: HT-Demucs (4 stems) + Spleeter (5 stems)
        all_results = list(htdemucs_results[:-1]) + list(spleeter_results[:-1])
        
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
        import traceback
        traceback.print_exc()
        return [None] * 11, f"❌ Error: {str(e)}"

# --- Gradio UI ---
print("Creating Gradio interface...")
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎵 Music Stem Separator - HT-Demucs & Spleeter 2025
    
    Upload your music and get stems from both **HT-Demucs** and **Spleeter** models!
    
    **HT-Demucs** provides: Drums, Bass, Other, Vocals  
    **Spleeter 2025** provides: Vocals, Drums, Bass, Other, **Piano** 🎹 (5stems model)
    
    Compare the quality and choose the best stems for your needs!
    """)

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(type="filepath", label="🎵 Upload Your Song")
            
            # Model selection toggles
            gr.Markdown("### 🎛️ Select Models to Run")
            with gr.Row():
                htdemucs_toggle = gr.Checkbox(label="🎯 HT-Demucs", value=True, info="Drums, Bass, Other, Vocals")
                spleeter_label = "🎵 Spleeter 2025 (5stems)" if spleeter_available else "🎵 Spleeter 2025"
                spleeter_info = "Vocals, Drums, Bass, Other, Piano" if spleeter_available else "Not available"
                spleeter_toggle = gr.Checkbox(
                    label=spleeter_label, 
                    value=spleeter_available, 
                    info=spleeter_info,
                    interactive=spleeter_available
                )
            
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
            gr.Markdown("### 🎵 Spleeter 2025 Results")
            with gr.Row():
                spleeter_vocals = gr.Audio(label="🎤 Vocals", type="filepath")
                spleeter_drums = gr.Audio(label="🥁 Drums", type="filepath")
            with gr.Row():
                spleeter_bass = gr.Audio(label="🎸 Bass", type="filepath")
                spleeter_other = gr.Audio(label="🎼 Other", type="filepath")
            with gr.Row():
                spleeter_piano = gr.Audio(label="🎹 Piano", type="filepath")
            
            if spleeter_available:
                gr.Markdown("*5stems model: Vocals, Drums, Bass, Other, Piano*")
            else:
                gr.Markdown("*Note: Spleeter 5stems model not available*")

    gr.Markdown("---")
    
    with gr.Row():
        comparison_text = f"""
        ### 📋 Model Comparison
        
        | Feature | HT-Demucs | Spleeter 2025 (5stems) |
        |---------|-----------|----------|
        | **Vocals** | ✅ High Quality | {'✅ Available' if spleeter_available else '❌ N/A'} |
        | **Drums** | ✅ High Quality | {'✅ Available' if spleeter_available else '❌ N/A'} |
        | **Bass** | ✅ High Quality | {'✅ Available' if spleeter_available else '❌ N/A'} |
        | **Other** | ✅ High Quality | {'✅ Available' if spleeter_available else '❌ N/A'} |
        | **Piano** | ❌ Not Available | {'✅ **Available**' if spleeter_available else '❌ N/A'} |
        | **Speed** | ⚡ Fast | {'⚡ Fast' if spleeter_available else '❌ N/A'} |
        | **Quality** | 🏆 Excellent | {'🏆 Good' if spleeter_available else '❌ N/A'} |
        
        **💡 Tip:** Use Spleeter 2025 for piano separation, HT-Demucs for other instruments!
        """
        gr.Markdown(comparison_text)

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
    🚀 Powered by <strong>HT-Demucs</strong> & <strong>Spleeter 2025</strong> | 
    🎵 Compare and choose your best stems!
    </p>
    """)

if __name__ == "__main__":
    demo.launch(share=True)