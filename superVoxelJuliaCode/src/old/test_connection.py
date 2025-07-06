#!/usr/bin/env python3
"""
PDF to Speech Converter using Bark TTS with German voices

This script takes a PDF file as input and converts its text to speech using 
various German voices available in the Bark text-to-speech system.
"""

import os
import sys
import argparse
from typing import List, Optional, Dict, Tuple
import time
import traceback

# Try importing required libraries, install if missing
try:
    import pypdf
    from scipy.io.wavfile import write as write_wav
except ImportError:
    print("Installing required dependencies...")
    os.system("pip install pypdf scipy")
    import pypdf
    from scipy.io.wavfile import write as write_wav

from bark import preload_models
from bark import SAMPLE_RATE, generate_audio


# Define German voice presets
GERMAN_VOICES = [
    "v2/de_speaker_3",
    "v2/de_speaker_4",
    "v2/de_speaker_5",
    "v2/de_speaker_6",
    "v2/de_speaker_7",
    "v2/de_speaker_8",
    "v2/de_speaker_9"
]

def install_bark():
    """Install Bark TTS if not already installed."""
    try:
        from bark import SAMPLE_RATE, generate_audio, preload_models
        print("Bark already installed.")
    except ImportError:
        print("Installing Bark TTS...")
        os.system("pip install git+https://github.com/suno-ai/bark.git")
        # Verify installation
        try:
            from bark import SAMPLE_RATE, generate_audio, preload_models
            print("Bark installed successfully.")
        except ImportError:
            print("Failed to install Bark. Please install it manually:")
            print("pip install git+https://github.com/suno-ai/bark.git")
            sys.exit(1)

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        String containing all text from the PDF
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    print(f"Reading PDF file: {pdf_path}")
    
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        total_pages = len(reader.pages)
        
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
            
            # Print progress for large PDFs
            if i % 10 == 0 or i == total_pages - 1:
                print(f"Processed {i+1}/{total_pages} pages...")
        
        # Basic stats
        word_count = len(text.split())
        print(f"Extracted {word_count} words from {total_pages} pages.")
        
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        raise

def split_into_chunks(text: str, max_chars: int = 200) -> List[str]:
    """
    Split text into smaller chunks suitable for TTS processing.
    
    Args:
        text: Text to split
        max_chars: Maximum characters per chunk
        
    Returns:
        List of text chunks
    """
    # Split by natural breakpoints (sentences)
    sentences = []
    for paragraph in text.split('\n'):
        # Split by sentence end markers and keep the markers
        for sentence in paragraph.replace('!', '!.').replace('?', '?.').split('.'):
            if sentence.strip():
                sentences.append(sentence.strip() + '.')
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence doesn't exceed limit, add it
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            # Add current chunk to results if not empty
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Handle very long sentences by splitting them
            if len(sentence) > max_chars:
                words = sentence.split()
                current_chunk = ""
                for word in words:
                    if len(current_chunk) + len(word) + 1 <= max_chars:
                        current_chunk += " " + word if current_chunk else word
                    else:
                        chunks.append(current_chunk.strip())
                        current_chunk = word
            else:
                current_chunk = sentence
    
    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    print(f"Split text into {len(chunks)} chunks (max {max_chars} chars per chunk)")
    return chunks

def generate_speech(text: str, voice_preset: str, output_file: str) -> None:
    """
    Generate speech from text using Bark TTS with a specific voice preset.
    
    Args:
        text: Text to convert to speech
        voice_preset: Bark voice preset to use
        output_file: Path to save the output WAV file
    """
    # Import here to avoid importing before potential installation
    
    print(f"Generating speech with voice '{voice_preset}'...")
    print(f"Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    
    start_time = time.time()
    
    # Generate audio
    audio_array = generate_audio(text, history_prompt=voice_preset)
    
    # Save to file
    write_wav(output_file, SAMPLE_RATE, audio_array)
    
    duration = time.time() - start_time
    audio_duration = len(audio_array) / SAMPLE_RATE
    print(f"Generated {audio_duration:.1f}s audio in {duration:.1f}s")
    print(f"Saved to: {output_file}")

def process_pdf(pdf_path: str, output_dir: str = "bark_output", 
                voices: Optional[List[str]] = None, max_chars: int = 200) -> None:
    """
    Process a PDF file and convert its text to speech using multiple German voices.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save audio files
        voices: List of voice presets to use (defaults to all German voices)
        max_chars: Maximum characters per text chunk
    """
    # Use all German voices if none specified
    if voices is None:
        voices = GERMAN_VOICES
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Import Bark after ensuring it's installed
    
    # Extract text from PDF
    try:
        text = extract_text_from_pdf(pdf_path)
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return
    
    # Split text into manageable chunks
    chunks = split_into_chunks(text, max_chars)
    
    # Write chunks to a reference file
    chunks_file = os.path.join(output_dir, "text_chunks.txt")
    with open(chunks_file, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"Chunk {i+1}:\n{chunk}\n\n")
    
    print(f"Saved text chunks to: {chunks_file}")
    
    # Preload Bark models (only needs to be done once)
    print("Preloading Bark models (this may take a while)...")
    preload_models()
    
    # Process each chunk with different voices
    success_count = 0
    error_count = 0
    
    for i, chunk in enumerate(chunks):
        # Select voice (rotate through available voices)
        voice_idx = i % len(voices)
        voice = voices[voice_idx]
        
        # Create output filename
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        safe_base_name = "".join(c if c.isalnum() else "_" for c in base_name)
        file_name = f"{safe_base_name}_chunk{i+1:03d}_{voice.replace('/', '_')}.wav"
        output_file = os.path.join(output_dir, file_name)
        
        try:
            # Generate speech
            generate_speech(chunk, voice, output_file)
            success_count += 1
        except Exception as e:
            print(f"Error generating speech for chunk {i+1}:")
            print(f"  {str(e)}")
            traceback.print_exc()
            error_count += 1
    
    print("\nSummary:")
    print(f"Successfully processed {success_count} chunks")
    if error_count > 0:
        print(f"Failed to process {error_count} chunks")
    print(f"Audio files saved to: {os.path.abspath(output_dir)}")

def main():
    """Main function handling command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert PDF text to speech using Bark TTS with German voices"
    )
    parser.add_argument(
        "pdf_path", 
        help="Path to the PDF file to convert"
    )
    parser.add_argument(
        "-o", "--output-dir", 
        default="bark_output",
        help="Directory to save audio files (default: bark_output)"
    )
    parser.add_argument(
        "-v", "--voices",
        help="Comma-separated list of voice presets to use (default: all German voices)"
    )
    parser.add_argument(
        "-m", "--max-chars",
        type=int, default=200,
        help="Maximum characters per chunk (default: 200)"
    )
    
    args = parser.parse_args()
    
    # Install Bark if needed
    install_bark()
    
    # Parse voice presets if provided
    voices = None
    if args.voices:
        voices = [v.strip() for v in args.voices.split(",")]
        for voice in voices:
            if not voice.startswith("v2/de_speaker_"):
                print(f"Warning: '{voice}' doesn't look like a standard German voice preset.")
    
    # Process the PDF
    process_pdf(args.pdf_path, args.output_dir, voices, args.max_chars)

if __name__ == "__main__":
    main()
