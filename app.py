import flask
from flask import Flask, request, render_template, send_file, jsonify
import torch
from torch.serialization import add_safe_globals
import os
import pkg_resources
from deep_translator import GoogleTranslator
import nltk
import re
import unicodedata
import time
import soundfile as sf
import librosa
import numpy as np
from pydub import AudioSegment
import uuid
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, filename="app.log", format="%(asctime)s - %(levelname)s - %(message)s")

# Download NLTK sentence tokenizer
try:
    nltk.download("punkt_tab", quiet=True)
except Exception as e:
    logging.error(f"Error downloading NLTK punkt_tab: {e}")
    raise

# Import and allowlist classes for safe unpickling
try:
    import TTS.tts.configs.xtts_config
    import TTS.tts.models.xtts
    import TTS.config.shared_configs
except ImportError as e:
    logging.error(f"Error importing TTS modules: {e}")
    print("Ensure Coqui TTS is installed: pip install TTS")
    raise

add_safe_globals([
    TTS.tts.configs.xtts_config.XttsConfig,
    TTS.tts.models.xtts.XttsAudioConfig,
    TTS.tts.models.xtts.XttsArgs,
    TTS.config.shared_configs.BaseDatasetConfig,
])

# Check Coqui TTS version
try:
    tts_version = pkg_resources.get_distribution("TTS").version
    logging.info(f"Coqui TTS version: {tts_version}")
except pkg_resources.DistributionNotFound:
    logging.warning("Coqui TTS not found. Installing latest version...")
    os.system("pip install TTS")

# Check and downgrade transformers version
try:
    import transformers
    transformers_version = transformers.__version__
    if transformers_version != "4.49.0":
        logging.warning(f"Transformers version {transformers_version} detected. Downgrading to 4.49.0.")
        os.system("pip uninstall transformers -y")
        os.system("pip install transformers==4.49.0")
except ImportError:
    logging.warning("Installing transformers 4.49.0...")
    os.system("pip install transformers==4.49.0")

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# Initialize TTS model
try:
    from TTS.api import TTS
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(device)
except Exception as e:
    logging.error(f"Error loading TTS model: {e}")
    raise

# Supported languages by Coqui TTS
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "zh-cn": "Chinese",
    "ja": "Japanese",
    "ko": "Korean"
}

# Create uploads and outputs directories
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

def preprocess_wav(input_wav, output_wav):
    try:
        audio, sr = librosa.load(input_wav, sr=22050, mono=True)
        audio, _ = librosa.effects.trim(audio, top_db=30)
        audio = audio / np.max(np.abs(audio)) * 0.8
        audio = librosa.effects.preemphasis(audio, coef=0.98)
        sf.write(output_wav, audio, sr, subtype="PCM_16")
        logging.info(f"Preprocessed {input_wav} to {output_wav}")
        return output_wav
    except Exception as e:
        logging.error(f"Error preprocessing {input_wav}: {e}")
        return input_wav

def translate_text_deep_translator(text, src="en", dest="hi"):
    max_retries = 3
    retry_delay = 2  # seconds
    for attempt in range(max_retries):
        try:
            translator = GoogleTranslator(source=src, target=dest)
            translation = translator.translate(text)
            if not translation:
                raise ValueError("Translation returned empty result")
            logging.info(f"Successfully translated text to {dest}: {translation[:50]}...")
            return translation
        except Exception as e:
            logging.warning(f"Translation attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise Exception(f"Translation failed after {max_retries} attempts: {str(e)}")

def split_text_by_tokens(text, max_chars=200):
    # Normalize text
    text = unicodedata.normalize("NFKC", text).strip()
    
    # Split by Hindi full stop ('।') or other sentence boundaries
    sentences = re.split(r'(।|\.|\?|!)\s*', text)
    # Combine punctuation with the preceding sentence
    sentences = [sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '') 
                 for i in range(0, len(sentences)-1, 2)] + ([sentences[-1]] if len(sentences) % 2 else [])
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = ""
    current_chars = 0
    
    for sentence in sentences:
        sentence_chars = len(unicodedata.normalize("NFKC", sentence))
        
        if current_chars + sentence_chars <= max_chars:
            current_chunk += sentence + " "
            current_chars += sentence_chars + 1
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
            current_chars = sentence_chars + 1
            
            # If a single sentence exceeds max_chars, split at the nearest full stop
            while current_chars > max_chars:
                # Find the nearest full stop ('।' or '.') before max_chars
                split_pos = max(current_chunk.rfind("।", 0, max_chars),
                              current_chunk.rfind(".", 0, max_chars))
                if split_pos == -1:
                    # If no full stop, fall back to space or hard split
                    split_pos = current_chunk.rfind(" ", 0, max_chars)
                    if split_pos == -1:
                        split_pos = max_chars  # Hard split if no natural boundary
                chunks.append(current_chunk[:split_pos + 1].strip())
                current_chunk = current_chunk[split_pos + 1:].strip()
                current_chars = len(unicodedata.normalize("NFKC", current_chunk))
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Filter out empty chunks and ensure final chunks are within limit
    final_chunks = []
    for chunk in chunks:
        char_count = len(unicodedata.normalize("NFKC", chunk))
        if char_count > max_chars:
            # Split oversized chunk at nearest full stop
            while char_count > max_chars:
                split_pos = max(chunk.rfind("।", 0, max_chars),
                              chunk.rfind(".", 0, max_chars))
                if split_pos == -1:
                    split_pos = chunk.rfind(" ", 0, max_chars)
                    if split_pos == -1:
                        split_pos = max_chars
                final_chunks.append(chunk[:split_pos + 1].strip())
                chunk = chunk[split_pos + 1:].strip()
                char_count = len(unicodedata.normalize("NFKC", chunk))
            if chunk.strip():
                final_chunks.append(chunk.strip())
        else:
            final_chunks.append(chunk)
    
    # Log chunk details
    for i, chunk in enumerate(final_chunks):
        logging.info(f"Chunk {i+1} ({len(unicodedata.normalize('NFKC', chunk))} chars): {chunk}")
    
    return final_chunks

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_text = request.form.get("input_text")
        target_language = request.form.get("target_language")
        audio_file = request.files.get("audio_file")

        if not input_text or not target_language or not audio_file:
            logging.error("Missing input: text, target_language, or audio_file")
            return jsonify({"error": "Please provide text, target language, and audio file."}), 400

        # Save and preprocess audio file
        audio_filename = f"{uuid.uuid4()}.wav"
        audio_path = os.path.join(app.config["UPLOAD_FOLDER"], audio_filename)
        audio_file.save(audio_path)
        processed_audio_path = os.path.join(app.config["UPLOAD_FOLDER"], f"processed_{audio_filename}")
        processed_audio_path = preprocess_wav(audio_path, processed_audio_path)

        # Validate audio file
        try:
            audio, sr = librosa.load(processed_audio_path, sr=22050, mono=True)
            if len(audio) == 0:
                raise ValueError("Processed audio file is empty or invalid")
            logging.info(f"Validated audio file {processed_audio_path}")
        except Exception as e:
            logging.error(f"Invalid audio file {processed_audio_path}: {str(e)}")
            if os.path.exists(audio_path):
                os.remove(audio_path)
            if os.path.exists(processed_audio_path):
                os.remove(processed_audio_path)
            return jsonify({"error": f"Invalid audio file: {str(e)}"}), 400

        # Translate text with retry mechanism
        try:
            translated_text = translate_text_deep_translator(input_text, src="en", dest=target_language)
            if not translated_text.strip():
                raise ValueError("Translated text is empty")
            logging.info(f"Translated text: {translated_text}")
        except Exception as e:
            logging.error(f"Translation failed: {str(e)}")
            if os.path.exists(audio_path):
                os.remove(audio_path)
            if os.path.exists(processed_audio_path):
                os.remove(processed_audio_path)
            return jsonify({"error": f"Translation failed: Unable to connect to translation service. Please check your internet connection or try again later."}), 503

        # Check text length and warn if exceeding limit
        char_count = len(unicodedata.normalize("NFKC", translated_text))
        if char_count > 200:
            logging.warning(f"The text length ({char_count} chars) exceeds the character limit of 200 for language '{target_language}', splitting into smaller chunks.")

        # Split translated text
        try:
            translated_sentences = split_text_by_tokens(translated_text, max_chars=200)
            if not translated_sentences:
                raise ValueError("No valid text chunks generated after splitting")
            logging.info(f"Split text into {len(translated_sentences)} chunks")
        except Exception as e:
            logging.error(f"Text splitting failed: {str(e)}")
            if os.path.exists(audio_path):
                os.remove(audio_path)
            if os.path.exists(processed_audio_path):
                os.remove(processed_audio_path)
            return jsonify({"error": f"Text splitting failed: {str(e)}"}), 500

        # Generate TTS for one chunk at a time and collect audio files
        temp_files = []
        output_filename = f"output_{uuid.uuid4()}.wav"
        output_path = os.path.join(app.config["OUTPUT_FOLDER"], output_filename)
        try:
            for i, chunk in enumerate(translated_sentences):
                if not chunk.strip():
                    logging.warning(f"Skipping empty chunk {i+1}")
                    continue
                char_count = len(unicodedata.normalize("NFKC", chunk))
                logging.info(f"Processing chunk {i+1} ({char_count} chars): {chunk[:50]}...")
                temp_output = os.path.join(app.config["OUTPUT_FOLDER"], f"temp_{uuid.uuid4()}.wav")
                try:
                    # Generate TTS for one chunk
                    tts.tts_to_file(
                        text=chunk,
                        speaker_wav=processed_audio_path,
                        language=target_language,
                        file_path=temp_output,
                        temperature=0.55,
                        top_p=0.7,
                        repetition_penalty=1.8,
                        length_penalty=0.8
                    )
                    # Verify the generated audio file
                    audio_segment = AudioSegment.from_wav(temp_output)
                    if audio_segment.duration_seconds > 0:
                        temp_files.append(temp_output)
                        logging.info(f"Successfully generated audio for chunk {i+1} at {temp_output}")
                    else:
                        logging.warning(f"Generated audio for chunk {i+1} is empty, skipping")
                        os.remove(temp_output)
                except Exception as e:
                    logging.warning(f"TTS failed for chunk {i+1}: {str(e)}")
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
                    continue

            if not temp_files:
                raise ValueError("No audio files generated. All chunks were invalid or skipped.")

            # Concatenate all audio files
            logging.info(f"Concatenating {len(temp_files)} audio files")
            combined = AudioSegment.empty()
            for i, temp_file in enumerate(temp_files):
                try:
                    audio = AudioSegment.from_wav(temp_file)
                    combined += audio
                    logging.info(f"Added chunk {i+1} audio to combined output")
                    os.remove(temp_file)  # Clean up temp file
                except Exception as e:
                    logging.warning(f"Failed to concatenate audio from {temp_file}: {str(e)}")
                    continue

            if combined.duration_seconds == 0:
                raise ValueError("Concatenated audio is empty")

            combined.export(output_path, format="wav")
            logging.info(f"Concatenated audio saved to {output_path}")

            # Clean up uploaded audio
            if os.path.exists(audio_path):
                os.remove(audio_path)
            if os.path.exists(processed_audio_path):
                os.remove(processed_audio_path)

            return jsonify({
                "translated_text": translated_text,
                "audio_url": f"/download/{output_filename}"
            })
        except Exception as e:
            logging.error(f"TTS generation or concatenation failed: {str(e)}")
            # Clean up any temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            if os.path.exists(audio_path):
                os.remove(audio_path)
            if os.path.exists(processed_audio_path):
                os.remove(processed_audio_path)
            return jsonify({"error": f"TTS generation or concatenation failed: {str(e)}"}), 500

    return render_template("index.html", languages=SUPPORTED_LANGUAGES)

@app.route("/download/<filename>")
def download(filename):
    return send_file(os.path.join(app.config["OUTPUT_FOLDER"], filename), as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)