# YouTube Music PCA Visualizer

A project I made to visualize the "semantic space" of songs using audio embeddings. Hover over points to hear the music and see which ones are close to each other.

## How it works
1. Downloads audio from YouTube URLs.
2. Uses the **MERT** model to extract musical features (embeddings).
3. Applies **PCA** to reduce them to 2D.
4. Generates an **interactive HTML plot** where you can hover to play song snippets.

## Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run the script
python yt_audio_pca.py "YOUR_YOUTUBE_URL_OR_PLAYLIST"

# If you get "Sign in to confirm you're not a bot" error:
python yt_audio_pca.py "YOUR_URL" --cookies cookies.txt
```

## How to get cookies.txt
Install a browser extension like "Get cookies.txt LOCALLY" (Chrome) or "cookies.txt" (Firefox), export your YouTube cookies as a `.txt` file, and save it in the project directory as `cookies.txt`.

## Viewing the Plot

Just open the HTML file.

---
**Disclaimer**: This was made just for fun/educational purposes. Don't use it to do illegal stuff. Respect YouTube's ToS and artists' copyrights. All audio files are ignored by git.
