import argparse
import os
import sys
import tempfile
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Model
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import yt_dlp
from tqdm import tqdm
import librosa

# Constants
MODEL_NAME = "m-a-p/MERT-v1-95M"
TARGET_SR = 24000
MAX_DURATION_SEC = 60  # Analyze up to 60 seconds to save time/memory

def get_args():
    parser = argparse.ArgumentParser(description="Generate Audio Embeddings from YouTube songs and plot them.")
    parser.add_argument("urls", nargs="+", help="List of YouTube Video or Playlist URLs")
    return parser.parse_args()

def download_audio(url, output_dir):
    """
    Downloads audio from YouTube URL into the output directory.
    Skips if file already exists in download archive.
    Returns: list of (filepath, title)
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'quiet': True,
        'ignoreerrors': True,
        'no_warnings': True,
        'download_archive': os.path.join(output_dir, 'downloaded.txt'),
    }

    downloaded_files = []

    # Options for metadata extraction (ignore archive so we get all entries)
    ydl_opts_meta = ydl_opts.copy()
    if 'download_archive' in ydl_opts_meta:
        del ydl_opts_meta['download_archive']
    
    # 1. Extract metadata WITHOUT downloading to get filenames
    print("Fetching metadata...")
    with yt_dlp.YoutubeDL(ydl_opts_meta) as ydl_meta:
        try:
            info = ydl_meta.extract_info(url, download=False)
        except Exception as e:
            print(f"Error extracting metadata for {url}: {e}")
            return []

    # 2. Trigger download (with archive check)
    print("Ensuring files are downloaded (checking archive, this may take a moment)...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
         try:
             ydl.download([url])
         except Exception as e:
             print(f"Error downloading {url}: {e}")

    # 3. Resolve filenames using metadata
    # We use ydl_meta to prepare filename because it has the same config (mostly)
    # Actually we should use the one that matches how we downloaded?
    # Filenames shouldn't depend on download_archive.
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl: # Use main opts for filename prep just to be safe
        if 'entries' in info:
            # Playlist
            entries = list(info['entries'])
            for entry in entries:
                if entry:
                    # prepare_filename might need the info dict to be "complete"
                    # entry from extract_info(download=False) should be enough
                    original_path = ydl.prepare_filename(entry)
                    base, _ = os.path.splitext(original_path)
                    filepath = f"{base}.wav"
                    if not os.path.isabs(filepath) and not filepath.startswith(output_dir):
                            filepath = os.path.join(output_dir, os.path.basename(filepath))
                    
                    downloaded_files.append((filepath, entry['title']))
        else:
            # Single video
            original_path = ydl.prepare_filename(info)
            base, _ = os.path.splitext(original_path)
            filepath = f"{base}.wav"
            if not os.path.isabs(filepath) and not filepath.startswith(output_dir):
                    filepath = os.path.join(output_dir, os.path.basename(filepath))
            
            downloaded_files.append((filepath, info['title']))

    return downloaded_files

def load_and_preprocess_audio(filepath):
    """
    Loads audio, resamples to 24kHz, and takes a segment from the middle.
    """
    try:
        # Load with librosa, automatically resamples to TARGET_SR
        # mono=True mixes to mono
        waveform, _ = librosa.load(filepath, sr=TARGET_SR, mono=True)
        
        # Convert to tensor [1, time]
        waveform = torch.tensor(waveform).unsqueeze(0)
        
        # Truncate/Crop
        num_frames = waveform.shape[1]
        max_frames = TARGET_SR * MAX_DURATION_SEC
        
        if num_frames > max_frames:
            # Take middle segment
            start = (num_frames - max_frames) // 2
            waveform = waveform[:, start:start+max_frames]
            
        return waveform, original_duration
    except Exception as e:
        # Expected if file doesn't exist or is corrupted
        # print(f"Error processing {filepath}: {e}")
        return None, 0

def generate_embedding(model, waveform):
    """
    Generates embedding using MERT model.
    """
    with torch.no_grad():
        if torch.cuda.is_available():
            input_values = waveform.cuda()
        else:
            input_values = waveform
            
        outputs = model(input_values, output_hidden_states=True)
        
        # MERT-v1-95M has 13 layers (0-12). 
        # Layer 12 (final) is good for audio features.
        
        last_hidden_state = outputs.last_hidden_state # [batch, time, 768]
        
        # Average over time dimension to get a single vector per song
        embedding = torch.mean(last_hidden_state, dim=1) # [batch, 768]
        
    return embedding.cpu().numpy().flatten()

import csv
import pandas as pd
import plotly.express as px

# ... (Previous constants)
CACHE_FILE = "embeddings_cache.csv"

def load_cache():
    """
    Loads embeddings from CSV cache.
    Returns: dict {title: embedding_numpy_array}
    """
    cache = {}
    if not os.path.exists(CACHE_FILE):
        return cache
        
    try:
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2: continue
                title = row[0]
                try:
                    embedding = np.array([float(x) for x in row[1:]])
                    cache[title] = embedding
                except ValueError:
                    continue
    except Exception as e:
        print(f"Error loading cache: {e}")
        
    return cache

def save_to_cache(title, embedding):
    """
    Appends a single embedding to the CSV cache.
    """
    try:
        with open(CACHE_FILE, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            row = [title] + embedding.tolist()
            writer.writerow(row)
    except Exception as e:
        print(f"Error saving to cache: {e}")

def main():
    args = get_args()

    # 1. Prepare Audio Directory
    audio_dir = os.path.join(os.getcwd(), "audio")
    os.makedirs(audio_dir, exist_ok=True)
    print(f"Audio directory: {audio_dir}")
    
    # Load Cache
    print(f"Loading embeddings cache from {CACHE_FILE}...")
    embedding_cache = load_cache()
    print(f"Loaded {len(embedding_cache)} embeddings from cache.")

    # 2. Download/Collect Audio Files
    print("Checking and downloading songs...")
    files_to_process = []
    
    for url in args.urls:
        files = download_audio(url, audio_dir)
        files_to_process.extend(files)
    
    valid_files = [f for f in files_to_process if os.path.exists(f[0])]
    print(f"Found {len(valid_files)} valid audio files to process.")
    
    if not valid_files:
        print("No audio files found/downloaded. Exiting.")
        sys.exit(0)
    
    # Check which files strictly need processing (not in cache)
    files_needing_inference = []
    all_embeddings = []
    all_titles = []
    all_filepaths = []
    all_durations = [] # We need to re-read duration even for cached items if we want accurate seeking
    
    # To avoid re-loading audio for cached embeddings just to get duration, 
    # we could:
    # 1. Store duration in cache (breaking change)
    # 2. Quickly read duration using librosa.get_duration or similar (fast)
    # 3. Just assume a value or load it. 
    # Let's use librosa.get_duration for all files, it's fast.
    
    print("Gathering metadata for all files...")
    file_metadata = {} # filepath -> duration
    
    for filepath, title in tqdm(valid_files, desc="Metadata"):
         try:
             # Just get duration, much faster than loading waveform
             duration = librosa.get_duration(path=filepath)
             file_metadata[filepath] = duration
         except Exception:
             file_metadata[filepath] = 0.0

    files_needing_inference = []
    
    for filepath, title in valid_files:
        if title in embedding_cache:
            all_embeddings.append(embedding_cache[title])
            all_titles.append(title)
            all_filepaths.append(filepath)
            all_durations.append(file_metadata.get(filepath, 0.0))
        else:
            files_needing_inference.append((filepath, title))
            
    # 3. Load Model (Only if we have new files to process)
    if files_needing_inference:
        print(f"Need to process {len(files_needing_inference)} new songs. Loading {MODEL_NAME}...")
        try:
            model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
            if torch.cuda.is_available():
                model = model.cuda()
            model.eval()
            
            print(f"Generating embeddings for new songs...")
            for filepath, title in tqdm(files_needing_inference):
                waveform, _ = load_and_preprocess_audio(filepath)
                if waveform is not None:
                    embedding = generate_embedding(model, waveform)
                    
                    # Add to current run lists
                    all_embeddings.append(embedding)
                    all_titles.append(title)
                    all_filepaths.append(filepath)
                    all_durations.append(file_metadata.get(filepath, 0.0))
                    
                    # Save to cache immediately
                    save_to_cache(title, embedding)
                    
        except Exception as e:
            print(f"Failed to load model or generate embeddings: {e}")
            sys.exit(1)
    else:
        print("All songs found in cache. Skipping model loading.")

    if not all_embeddings:
        print("No embeddings generated.")
        sys.exit(1)

    # PCA
    X = np.array(all_embeddings)
    if len(X) < 2:
        print("Need at least 2 songs for PCA.")
        sys.exit(1)
        
    print("Performing PCA...")
    pca = PCA(n_components=2)
    X_r = pca.fit_transform(X)

    # Prepare Data for Plotly
    # We need relative paths for the browser to load them via a local server
    # absolute paths like /home/user/... usually blocked by browsers unless file://
    # Best practice: use relative path from where the html is served
    # The HTML will be in the project root. 'audio/' is in project root.
    # So relative path is just 'audio/filename.wav'
    
    relative_paths = []
    for fp in all_filepaths:
        try:
            rel = os.path.relpath(fp, os.getcwd())
            relative_paths.append(rel)
        except ValueError:
            relative_paths.append(fp)

    df = pd.DataFrame({
        'x': X_r[:, 0],
        'y': X_r[:, 1],
        'title': all_titles,
        'filepath': relative_paths,
        'duration': all_durations
    })
    
    print("Generating Interactive Plot...")
    
    fig = px.scatter(df, x='x', y='y', hover_name='title', 
                     custom_data=['filepath', 'duration'],
                     title='Song Embeddings PCA (Audio on Hover)',
                     template='plotly_dark')
    
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(width=1200, height=1000)

    # Custom JS for Audio Playback on Hover
    # We use a single global audio element to avoid overlap
    
    custom_js = """
    <script>
        var audio = new Audio();
        var currentSrc = "";
        
        // Listen to Plotly hover event
        var plot = document.getElementById('plotly-graph-div');
        
        // Polling for the plot div to be ready might be needed in some contexts, 
        // but typically embedding the script after the div works.
        // Plotly usually assigns the id 'plotly-graph-div' to the first plot container if not specified.
        // However, plotly.io.write_html generates a guid-based id.
        // We need a robust way to attach.
        // The safest way with independent HTML is to wait for DOMContentLoaded and find the collection.
        
        document.addEventListener("DOMContentLoaded", function(){
            var plotDivs = document.getElementsByClassName('plotly-graph-div');
            if(plotDivs.length > 0){
                var myPlot = plotDivs[0];
                
                myPlot.on('plotly_hover', function(data){
                    var point = data.points[0];
                    var filepath = point.customdata[0];
                    var duration = point.customdata[1];
                    
                    if(currentSrc !== filepath){
                        currentSrc = filepath;
                        audio.src = filepath;
                        audio.loop = true;
                        // Seek to middle
                        if(duration > 0){
                             audio.currentTime = duration / 2;
                        }
                        audio.play().catch(e => console.log("Audio play failed:", e));
                    } else {
                        audio.play().catch(e => console.log("Audio play failed:", e));
                    }
                });
                
                myPlot.on('plotly_unhover', function(data){
                    audio.pause();
                });
            }
        });
    </script>
    """
    
    output_html = "song_embeddings_pca.html"
    
    # Save with full html
    # Plotly's `write_html` allows passing `post_script` but injecting raw script tag via raw HTML manipulation is often surer for custom event listeners
    # But `write_html` with `include_plotlyjs='cdn'` results in a file we can read and append to.
    
    fig.write_html(output_html, include_plotlyjs='cdn', full_html=True)
    
    # Append JS manually before </body>
    with open(output_html, 'r', encoding='utf-8') as f:
        html_content = f.read()
        
    final_html = html_content.replace('</body>', f'{custom_js}\n</body>')
    
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(final_html)
        
    print(f"Interactive plot saved to {output_html}")
    print("NOTE: To view this with working audio, you must run a local server due to browser security policies.")
    print("Run: python3 -m http.server 8000")
    print("Then open: http://localhost:8000/song_embeddings_pca.html")

if __name__ == "__main__":
    main()
