# ./read_poem.py

# ---------------------------------- Imports ----------------------------------
from __future__ import annotations
from dotenv import load_dotenv
from openai import OpenAI
import os
from pathlib import Path
import socket
import subprocess
import sys

# ------------------------------------ Config ----------------------------------
# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent

ipaddress = socket.gethostbyname(socket.gethostname())
if (ipaddress == "192.168.0.102"):
    ENV_PATH = PARENT_DIR / "05.env"
else:
    ENV_PATH = SCRIPT_DIR / ".env"

# 🔑 LOAD the env file here
load_dotenv(ENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ------------------------------------ Constants / Variables ----------------------------------
POEM = """
Wrapped in merino's gentle embrace,
While winter's breath nips at my face.
Wool and leather, warm and true,
Through morning frost and skies of blue.
Professional comfort, styled with care,
Ready for whatever weather's there.
""".strip()


MODEL = "gpt-4o-mini-tts"
# VOICE = "cedar"
VOICE = "marin"

INSTRUCTIONS = (
    "Read this poem with a warm, calm, intimate tone. "
    "Speak clearly with gentle pauses at line breaks. "
    "Slightly slower pace, soothing and confident."
)


OUT_FILE = Path(__file__).with_name("poem.mp3")


def open_audio_file(path: Path) -> None:
    """
    Open the audio file using the OS default player.
    """
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception as e:
        print(f"Could not auto-open audio file: {e}")
        print(f"Audio saved at: {path.resolve()}")


def main() -> None:
    """
    1) Create TTS audio from POEM
    2) Stream it to an MP3 file
    3) Open it with your default player
    """


    client = OpenAI(api_key=OPENAI_API_KEY)

    print(f"Generating speech with voice='{VOICE}' -> {OUT_FILE.name}")

    # Stream output straight to file
    with client.audio.speech.with_streaming_response.create(
        model=MODEL,
        voice=VOICE,
        input=POEM,
        instructions=INSTRUCTIONS,
        # response_format="mp3",  # mp3 is default; you can set wav/pcm too :contentReference[oaicite:4]{index=4}
    ) as response:
        response.stream_to_file(OUT_FILE)

    print("Done.")
    print(f"Saved: {OUT_FILE.resolve()}")

    # Optional: auto-open / play
    open_audio_file(OUT_FILE)


if __name__ == "__main__":
    main()
