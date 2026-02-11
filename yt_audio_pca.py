import argparse
import os
import sys
import numpy as np
import torch
from transformers import Wav2Vec2Model
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import yt_dlp
from tqdm import tqdm
import librosa
import static_ffmpeg
import psutil
static_ffmpeg.add_paths()

# Constants
MODEL_NAME = "m-a-p/MERT-v1-95M"
TARGET_SR = 24000
# MAX_DURATION_SEC removed to analyze full song

def get_args():
    parser = argparse.ArgumentParser(description="Generate Audio Embeddings from YouTube songs and plot them.")
    parser.add_argument("urls", nargs="*", help="List of YouTube Video or Playlist URLs")
    parser.add_argument("--cookies", help="Path to cookies.txt file for authentication")
    parser.add_argument("--local", action="store_true", help="Skip YouTube entirely and process all files in the audio/ directory")
    return parser.parse_args()

def download_audio(url, output_dir, cookiefile=None):
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
    
    if cookiefile:
        ydl_opts['cookiefile'] = cookiefile

    downloaded_files = []

    # Options for metadata extraction (ignore archive so we get all entries)
    ydl_opts_meta = ydl_opts.copy()
    ydl_opts_meta['format'] = None  # Just get basic info, don't check formats
    if 'download_archive' in ydl_opts_meta:
        del ydl_opts_meta['download_archive']
    if 'postprocessors' in ydl_opts_meta:
        del ydl_opts_meta['postprocessors']
    
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
        
        original_duration = len(waveform) / TARGET_SR
        
        # Convert to tensor [1, time]
        waveform = torch.tensor(waveform).unsqueeze(0)
            
        return waveform, original_duration
    except Exception as e:
        # Expected if file doesn't exist or is corrupted
        # print(f"Error processing {filepath}: {e}")
        return None, 0

def generate_embedding(model, waveform):
    """
    Generates embedding using MERT model. 
    Dynamically chooses between full-song and chunked processing based on available RAM.
    """
    total_samples = waveform.shape[1]
    
    # Heuristic: Wav2Vec2 attention is roughly O(L^2).
    # Sequences longer than ~3-5 mins at 24kHz can be heavy.
    # We estimate based on available memory.
    
    available_gb = psutil.virtual_memory().available / (1024 ** 3)
    
    # 24kHz audio -> Downsampled ~320x -> ~75 frames per second
    # L = duration_sec * 75
    # Memory ~ L^2 * intermediate_layers * heads * size
    # Let's use a conservative threshold for "too long for one go"
    # Even with 256GB, very long songs (e.g. 1 hour) are impossible $O(L^2)$
    
    chunk_size_sec = 10
    max_full_song_sec = 300 # 5 mins default safe limit even for high RAM
    
    # Increase limit if we have massive RAM (256GB user request)
    if available_gb > 200:
        max_full_song_sec = 1800 # 30 mins
    elif available_gb > 64:
        max_full_song_sec = 600 # 10 mins
        
    duration_sec = total_samples / TARGET_SR
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if duration_sec <= max_full_song_sec:
        # Process the whole song at once
        try:
            with torch.no_grad():
                input_values = waveform.to(device)
                outputs = model(input_values, output_hidden_states=True)
                last_hidden_state = outputs.last_hidden_state
                embedding = torch.mean(last_hidden_state, dim=1).cpu().numpy().flatten()
                return embedding
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("GPU/CPU OOM detected for full song. Falling back to chunking...")
                # Fallthrough to chunking logic
            else:
                raise e

    # Chunking Strategy
    print(f"Analyzing in {chunk_size_sec}s chunks (Total: {duration_sec:.1f}s)...")
    chunk_size = chunk_size_sec * TARGET_SR
    all_chunk_embeddings = []
    
    with torch.no_grad():
        for i in range(0, total_samples, chunk_size):
            chunk = waveform[:, i:i+chunk_size].to(device)
            if chunk.shape[1] < TARGET_SR: continue # Skip fragments < 1s
                
            outputs = model(chunk, output_hidden_states=True)
            last_hidden_state = outputs.last_hidden_state
            chunk_embedding = torch.mean(last_hidden_state, dim=1).cpu()
            all_chunk_embeddings.append(chunk_embedding)
            
        if not all_chunk_embeddings: return None
        mean_embedding = torch.mean(torch.stack(all_chunk_embeddings), dim=0)
        
    return mean_embedding.numpy().flatten()

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
    files_to_process = []
    
    if args.local:
        print("Local mode: Scanning audio/ directory...")
        for file in os.listdir(audio_dir):
            if file.endswith(".wav"):
                filepath = os.path.join(audio_dir, file)
                title = os.path.splitext(file)[0]
                files_to_process.append((filepath, title))
    else:
        if not args.urls:
            print("Error: Provide at least one URL or use --local.")
            sys.exit(1)
        print("Checking and downloading songs from YouTube...")
        for url in args.urls:
            files = download_audio(url, audio_dir, cookiefile=args.cookies)
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

    # Prepare embeddings for analysis
    X = np.array(all_embeddings)
    if len(X) < 2:
        print("Need at least 2 songs for analysis.")
        sys.exit(1)

    # 4. Automatic Clustering (Gaussian Mixture Model with BIC)
    # We find the 'best' number of clusters by minimizing BIC on high-dim space
    print("Finding best clustering configuration (high-dim)...")
    n_samples = X.shape[0]
    max_k = min(11, n_samples)
    
    best_bic = np.inf
    best_gmm = None
    best_k = 1
    
    for k in range(1, max_k):
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(X)
        bic = gmm.bic(X)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_k = k
            
    print(f"Optimal clusters found: {best_k}")
    cluster_labels = best_gmm.predict(X)
    cluster_names = [f"Cluster {c}" for c in cluster_labels]

    # 5. Dimensionality Reduction (PCA)
    print("Performing PCA reduction to 2D...")
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
        'duration': all_durations,
        'cluster': cluster_names
    })
    
    print("Generating Interactive Plot...")
    
    fig = px.scatter(df, x='x', y='y', hover_name='title', 
                     color='cluster',
                     custom_data=['filepath', 'duration'],
                     title=f'Song Embeddings PCA (Audio on Hover) - {best_k} Clusters Found',
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
