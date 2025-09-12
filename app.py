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
import subprocess
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

# Setup Spleeter with command-line approach
print("Setting up Spleeter...")
spleeter_available = False

def check_spleeter_installation():
    """Check if Spleeter is installed and available via command line"""
    try:
        result = subprocess.run(['spleeter', '--help'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Spleeter command-line tool is available!")
            return True
        else:
            print(f"❌ Spleeter command failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ Spleeter command not found. Please install Spleeter.")
        return False
    except Exception as e:
        print(f"❌ Error checking Spleeter: {e}")
        return False

spleeter_available = check_spleeter_installation()

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
    Uses command-line execution for reliability.
    Returns FILE PATHS.
    """
    if audio_path is None:
        return None, None, None, None, None, "Please upload an audio file."

    if not spleeter_available:
        return None, None, None, None, None, "❌ Spleeter not available. Please install Spleeter."

    try:
        print(f"Spleeter: Processing audio from: {audio_path}")
        
        # Create output directory with timestamp
        timestamp = int(time.time() * 1000)
        output_dir = f"spleeter_stems_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Run Spleeter command-line tool
        cmd = [
            'spleeter', 'separate',
            '-i', audio_path,
            '-o', output_dir,
            '-p', 'spleeter:5stems-16kHz'
        ]
        
        print(f"Spleeter: Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"Spleeter command failed: {result.stderr}")
            return None, None, None, None, None, f"❌ Spleeter command failed: {result.stderr}"
        
        print("Spleeter: Separation complete.")
        
        # Find the separated files
        # Spleeter creates a subdirectory with the input filename
        input_filename = os.path.splitext(os.path.basename(audio_path))[0]
        spleeter_output_dir = os.path.join(output_dir, input_filename)
        
        if not os.path.exists(spleeter_output_dir):
            print(f"Expected output directory not found: {spleeter_output_dir}")
            return None, None, None, None, None, "❌ Spleeter output directory not found"
        
        # Map Spleeter output files to our expected order
        stem_mapping = {
            "vocals": "vocals.wav",
            "drums": "drums.wav", 
            "bass": "bass.wav",
            "other": "other.wav",
            "piano": "piano.wav"
        }
        
        output_paths = []
        for stem_name, filename in stem_mapping.items():
            source_path = os.path.join(spleeter_output_dir, filename)
            if os.path.exists(source_path):
                # Copy to our timestamped directory for consistency
                dest_path = os.path.join(output_dir, f"{stem_name}_{timestamp}.wav")
                shutil.copy2(source_path, dest_path)
                output_paths.append(dest_path)
                print(f"✅ Spleeter saved {stem_name} to {dest_path}")
            else:
                print(f"⚠️ Warning: {stem_name} file not found: {source_path}")
                output_paths.append(None)
        
        # Clean up the intermediate directory
        if os.path.exists(spleeter_output_dir):
            shutil.rmtree(spleeter_output_dir)
        
        return output_paths[0], output_paths[1], output_paths[2], output_paths[3], output_paths[4], "✅ Spleeter separation successful!"

    except subprocess.TimeoutExpired:
        return None, None, None, None, None, "❌ Spleeter separation timed out (5 minutes)"
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
                spleeter_toggle = gr.Checkbox(
                    label="🎵 Spleeter 2025 (5stems)", 
                    value=spleeter_available, 
                    info="Vocals, Drums, Bass, Other, Piano" if spleeter_available else "Not available",
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