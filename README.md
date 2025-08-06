# Text-to-Speech Translation Web Application

This is a Flask-based web application that translates text from English to a target language and generates text-to-speech (TTS) audio using a provided speaker voice. The application uses the Coqui TTS model (`xtts_v2`) for voice synthesis and supports multiple languages.

## Features
- Translate text from English to supported languages (e.g., Hindi, Spanish, French, etc.).
- Generate TTS audio using a provided WAV file as the speaker's voice.
- Preprocess audio to ensure compatibility with TTS.
- Split long text into chunks for processing.
- Serve generated audio files for download.

## Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster processing)
- Internet connection for translation services

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Prazeen7/Multilingual-Text-to-Speech-.git
   cd Multilingual-Text-to-Speech
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate 
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Flask application:
   ```bash
   python app.py
   ```
2. Open a web browser and navigate to `http://localhost:5000`.
3. Upload a WAV audio file, enter text, select a target language, and submit the form.
4. Download the generated audio file containing the translated text in the speaker's voice.

## Project Structure
- `app.py`: Main Flask application with routes for text translation and TTS generation.
- `uploads/`: Directory for storing uploaded audio files.
- `outputs/`: Directory for storing generated audio files.
- `templates/index.html`: HTML template for the web interface.
- `app.log`: Log file for application events and errors.

## Supported Languages
- English (en)
- Hindi (hi)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Portuguese (pt)
- Chinese (zh-cn)
- Japanese (ja)
- Korean (ko)

## Notes
- Ensure the uploaded WAV file is valid and contains clear speech for optimal TTS results.
- Text exceeding 200 characters will be split into chunks to comply with TTS limitations.
- The application uses `transformers==4.49.0` for compatibility with the Coqui TTS model.
- Logs are stored in `app.log` for debugging and monitoring.

## Troubleshooting
- **TTS model fails to load**: Ensure `TTS` is installed (`pip install TTS`) and check for compatible GPU drivers if using CUDA.
- **Translation errors**: Verify internet connectivity, as the app relies on Google Translate.
- **Audio issues**: Ensure uploaded WAV files are mono, 22kHz, and not empty.

## License
This project is licensed under the MIT License.
