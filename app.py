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
import librosa
import requests
import tarfile
import shutil
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

# Load Spleeter model with better error handling
print("Loading Spleeter model...")
spleeter_separator = None
spleeter_audio_adapter = None

try:
    # Set up environment variables for better model handling
    os.environ['SPLEETER_MODEL_PATH'] = '/tmp/spleeter_models'
    os.makedirs('/tmp/spleeter_models', exist_ok=True)
    
    # Try different approaches to handle the redirect issue
    import ssl
    import urllib.request
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # Create unverified SSL context to handle redirects
    ssl._create_default_https_context = ssl._create_unverified_context
    
    try:
        print("Attempting to load 5stems model...")
        # Try with specific configuration to handle redirects
        spleeter_separator = Separator(
            'spleeter:5stems', 
            multiprocess=False,
            stft_backend='tensorflow'
        )
        spleeter_model_type = "5stems"
        print("Spleeter: Using 5stems model (vocals, drums, bass, other, piano)")
    except Exception as e5:
        print(f"5stems model failed: {e5}")
        try:
            print("Attempting to load 2stems model...")
            spleeter_separator = Separator(
                'spleeter:2stems', 
                multiprocess=False,
                stft_backend='tensorflow'
            )
            spleeter_model_type = "2stems"
            print("Spleeter: Using 2stems model (vocals, accompaniment)")
        except Exception as e2:
            print(f"2stems model also failed: {e2}")
            try:
                print("Attempting to load 2stems-16kHz model...")
                spleeter_separator = Separator(
                    'spleeter:2stems-16kHz', 
                    multiprocess=False,
                    stft_backend='tensorflow'
                )
                spleeter_model_type = "2stems-16kHz"
                print("Spleeter: Using 2stems-16kHz model")
            except Exception as e3:
                print(f"All Spleeter models failed: {e3}")
                spleeter_separator = None
                spleeter_model_type = None
    
    if spleeter_separator is not None:
        spleeter_audio_adapter = AudioAdapter.default()
        print("Spleeter model loaded successfully.")
    else:
        print("Spleeter will be disabled for this session.")
        
except Exception as e:
    print(f"Spleeter initialization failed: {e}")
    spleeter_separator = None
    spleeter_audio_adapter = None
    spleeter_model_type = None

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
        
        # Use librosa for more robust audio loading
        try:
            # Load audio with librosa (handles more formats reliably)
            waveform, sample_rate = librosa.load(audio_path, sr=44100, mono=False)
            print(f"Spleeter (librosa): Loaded audio - shape: {waveform.shape}, sr: {sample_rate}")
            
            # Handle different audio shapes
            if waveform.ndim == 1:
                # Mono audio - convert to stereo for Spleeter
                print("Spleeter: Converting mono to stereo")
                waveform = np.stack([waveform, waveform], axis=0)
            elif waveform.ndim == 2 and waveform.shape[0] == 2:
                # Stereo audio - already correct format
                print("Spleeter: Stereo audio detected")
            else:
                print(f"Spleeter: Unexpected shape {waveform.shape}, converting...")
                if waveform.shape[0] > waveform.shape[1]:
                    # Transpose if needed (samples, channels) -> (channels, samples)
                    waveform = waveform.T
                if waveform.shape[0] == 1:
                    waveform = np.vstack([waveform, waveform])
                elif waveform.shape[0] > 2:
                    # Take first two channels if more than stereo
                    waveform = waveform[:2, :]
            
            print(f"Spleeter: Final waveform shape: {waveform.shape}")
            
            # Transpose to (samples, channels) format for Spleeter
            waveform_for_spleeter = waveform.T
            print(f"Spleeter: Transposed for separation - shape: {waveform_for_spleeter.shape}")
            
        except Exception as load_error:
            print(f"Librosa loading failed: {load_error}")
            # Fallback to spleeter's audio adapter
            waveform_for_spleeter, sample_rate = spleeter_audio_adapter.load(audio_path)
            print(f"Spleeter (adapter): Loaded audio - shape: {waveform_for_spleeter.shape}, sr: {sample_rate}")
        
        print("Spleeter: Applying the separation model...")
        # Use the waveform directly with Spleeter
        prediction = spleeter_separator.separate(waveform_for_spleeter)
        print("Spleeter: Separation complete.")
        print(f"Spleeter: Prediction keys: {list(prediction.keys())}")

        # Save stems temporarily
        output_dir = "spleeter_stems"
        os.makedirs(output_dir, exist_ok=True)

        output_paths = []
        
        # Handle different model types
        if spleeter_model_type == "5stems":
            # 5stems model
            stem_names = ["vocals", "drums", "bass", "other", "piano"]
        else:
            # 2stems model
            stem_names = ["vocals", "accompaniment", None, None, None]
        
        for i, name in enumerate(stem_names):
            if name is not None and name in prediction:
                out_path = os.path.join(output_dir, f"{name}.wav")
                stem_audio = prediction[name]
                
                print(f"Spleeter: {name} audio shape: {stem_audio.shape}, dtype: {stem_audio.dtype}")
                
                # Ensure correct format for saving
                if stem_audio.ndim == 1:
                    # Mono - reshape to (samples, 1)
                    stem_audio = stem_audio.reshape(-1, 1)
                elif stem_audio.ndim == 2:
                    # Check if it's (channels, samples) and transpose if needed
                    if stem_audio.shape[0] < stem_audio.shape[1] and stem_audio.shape[0] <= 2:
                        stem_audio = stem_audio.T
                
                # Save using soundfile for better compatibility
                sf.write(out_path, stem_audio, sample_rate)
                output_paths.append(out_path)
                print(f"✅ Spleeter saved {name} to {out_path}")
            else:
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
    # 🎵 Music Stem Separator - HT-Demucs & Spleeter Comparison
    
    Upload your music and get stems from both **HT-Demucs** and **Spleeter** models!
    
    **HT-Demucs** provides: Drums, Bass, Other, Vocals  
    **Spleeter** provides: Vocals, Drums, Bass, Other, **Piano** 🎹 (5stems model)
    
    Compare the quality and choose the best stems for your needs!
    """)

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(type="filepath", label="🎵 Upload Your Song")
            
            # Model selection toggles
            gr.Markdown("### 🎛️ Select Models to Run")
            with gr.Row():
                htdemucs_toggle = gr.Checkbox(label="🎯 HT-Demucs", value=True, info="Drums, Bass, Other, Vocals")
                spleeter_enabled = spleeter_separator is not None
                spleeter_toggle = gr.Checkbox(
                    label="🎵 Spleeter", 
                    value=spleeter_enabled, 
                    info=f"Available: {spleeter_model_type}" if spleeter_enabled else "Not available",
                    interactive=spleeter_enabled
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
            gr.Markdown("### 🎵 Spleeter Results")
            with gr.Row():
                spleeter_vocals = gr.Audio(label="🎤 Vocals", type="filepath")
                spleeter_drums = gr.Audio(label="🥁 Drums", type="filepath")
            with gr.Row():
                spleeter_bass = gr.Audio(label="🎸 Bass", type="filepath")
                spleeter_other = gr.Audio(label="🎼 Other", type="filepath")
            with gr.Row():
                spleeter_piano = gr.Audio(label="🎹 Piano", type="filepath")
            
            if spleeter_model_type == "2stems":
                gr.Markdown("*Note: Only Vocals and Accompaniment available with 2stems model*")
            elif not spleeter_enabled:
                gr.Markdown("*Note: Spleeter model not available*")

    gr.Markdown("---")
    
    with gr.Row():
        comparison_text = f"""
        ### 📋 Model Comparison
        
        | Feature | HT-Demucs | Spleeter ({spleeter_model_type if spleeter_model_type else 'N/A'}) |
        |---------|-----------|----------|
        | **Vocals** | ✅ High Quality | {'✅ Available' if spleeter_enabled else '❌ N/A'} |
        | **Drums** | ✅ High Quality | {'✅ Available' if spleeter_model_type == '5stems' else '❌ N/A'} |
        | **Bass** | ✅ High Quality | {'✅ Available' if spleeter_model_type == '5stems' else '❌ N/A'} |
        | **Other** | ✅ High Quality | {'✅ Available' if spleeter_model_type == '5stems' else '❌ N/A'} |
        | **Piano** | ❌ Not Available | {'✅ Available' if spleeter_model_type == '5stems' else '❌ N/A'} |
        | **Accompaniment** | ❌ Not Available | {'✅ Available' if spleeter_model_type == '2stems' else '❌ N/A'} |
        | **Speed** | ⚡ Fast | {'⚡ Fast' if spleeter_enabled else '❌ N/A'} |
        | **Quality** | 🏆 Excellent | {'🏆 Good' if spleeter_enabled else '❌ N/A'} |
        
        **💡 Tip:** Use Spleeter when you need piano separation, HT-Demucs for other instruments!
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
    🚀 Powered by <strong>HT-Demucs</strong> & <strong>Spleeter</strong> | 
    🎵 Compare and choose your best stems!
    </p>
    """)

if __name__ == "__main__":
    demo.launch(share=True)