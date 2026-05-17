cat > subtitles/gemini_client.py <<'EOF'
import os
from dotenv import load_dotenv


load_dotenv()


class GeminiSubtitleClient:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")

        self.api_key = api_key

    def generate_subtitles(self, audio_path, language_hint="ko-KR"):
        raise NotImplementedError("Subtitle generation is not implemented yet.")
EOF