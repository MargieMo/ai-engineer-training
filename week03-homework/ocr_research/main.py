import os
import json
import base64
import mimetypes
from pathlib import Path
from typing import List, Union, Optional

from dotenv import load_dotenv
from openai import OpenAI

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


# Prompt that asks the vision model to act as an OCR engine and return structured JSON.
OCR_PROMPT = (
    "You are an OCR engine. Extract ALL text visible in the image.\n"
    "Return STRICT JSON with this schema and nothing else:\n"
    '{"language": "<ISO code like en/zh>", '
    '"blocks": [{"text": "<one logical text block>", "confidence": <float 0-1>}]}\n'
    "Rules:\n"
    "- One array element per logical text block (a line, label, button, or heading).\n"
    "- Preserve the original characters/casing; do not translate.\n"
    "- `confidence` is your own estimate of how sure you are about that block.\n"
    "- If the image contains no text, return an empty `blocks` array."
)


class ImageOCRReader(BaseReader):
    """Extract text from images using a GPT-5 multimodal vision model and return Documents.

    Unlike a classic PaddleOCR-based reader, this implementation sends each image to an
    OpenAI vision model (default ``gpt-5``) and asks it to transcribe the text. The model
    returns structured blocks, which are concatenated into the Document text and summarized
    in the Document metadata.
    """

    def __init__(
        self,
        model: str = "gpt-5",
        lang: str = "auto",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        """
        Args:
            model: OpenAI vision-capable model name used for OCR.
            lang: Hint for the expected language ('auto', 'en', 'ch', ...). Informational only;
                the model auto-detects and reports the detected language in metadata.
            api_key: OpenAI API key. Falls back to the OPENAI_API_KEY environment variable.
            api_base: OpenAI API base URL. Falls back to OPENAI_API_BASE or the public endpoint.
        """
        super().__init__()
        self.model = model
        self.lang = lang
        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=api_base or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        )

    def _encode_image(self, image_path: str) -> str:
        """Read an image file and return a base64 data URI suitable for the vision API."""
        mime, _ = mimetypes.guess_type(image_path)
        mime = mime or "image/png"
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    def _ocr_one(self, image_path: str) -> dict:
        """Call the vision model on a single image and return the parsed JSON result."""
        data_uri = self._encode_image(image_path)
        resp = self._client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": OCR_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
        )
        content = resp.choices[0].message.content or "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Fallback: treat the whole response as a single text block
            return {"language": self.lang, "blocks": [{"text": content.strip(), "confidence": 0.0}]}

    def load_data(self, file: Union[str, Path, List[Union[str, Path]]]) -> List[Document]:
        """Extract text from one or more image files and return a list of Documents.

        Args:
            file: A single image path or a list of image paths.

        Returns:
            A list of Document objects, one per image, each carrying OCR metadata
            (image_path, ocr_model, language, num_text_blocks, avg_confidence).
        """
        files = [file] if isinstance(file, (str, Path)) else file

        documents: List[Document] = []
        for image_path in files:
            image_path_str = str(image_path)
            result = self._ocr_one(image_path_str)

            blocks = result.get("blocks", []) or []
            if not blocks:
                print(f"Warning: No text detected in {image_path_str}")
                continue

            text_blocks = []
            confidences = []
            for i, block in enumerate(blocks):
                text = (block.get("text") or "").strip()
                conf = float(block.get("confidence", 0.0) or 0.0)
                text_blocks.append(f"[Text Block {i + 1}] (conf: {conf:.2f}): {text}")
                confidences.append(conf)

            full_text = "\n".join(text_blocks)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            doc = Document(
                text=full_text,
                metadata={
                    "image_path": image_path_str,
                    "ocr_model": self.model,
                    "language": result.get("language", self.lang),
                    "num_text_blocks": len(text_blocks),
                    "avg_confidence": round(float(avg_confidence), 4),
                },
            )
            documents.append(doc)

        return documents

    def load_data_from_dir(self, dir_path: Union[str, Path]) -> List[Document]:
        """Convenience helper: OCR every image in a directory (batch processing)."""
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        image_files = sorted(
            p for p in Path(dir_path).glob("*") if p.suffix.lower() in exts
        )
        return self.load_data(image_files)


def setup_environment():
    """Configure LlamaIndex to use GPT-5 as the LLM and OpenAI embeddings."""
    from llama_index.core import Settings
    from llama_index.llms.openai import OpenAI as LlamaOpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding

    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

    # Note: gpt-5 is a reasoning model that only supports the default temperature.
    Settings.llm = LlamaOpenAI(model="gpt-5", api_key=api_key, api_base=api_base)
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small", api_key=api_key, api_base=api_base
    )
    print("LlamaIndex environment setup complete.")


def main():
    # The assignment entry point goes here. Run from the project root with:
    #   python -m ocr_research.main
    load_dotenv()

    # --- 1. Locate the sample images (reused from homework_examples) ---
    print("--- Step 1: Locating sample images ---")
    data_dir = Path("data/ocr_images")
    image_files = sorted(
        p for p in data_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not image_files:
        print(f"No images found in {data_dir.resolve()}. Exiting.")
        return
    print(f"Found {len(image_files)} images in {data_dir.resolve()}")

    # --- 2. Run OCR with the GPT-5 based ImageOCRReader ---
    print("\n--- Step 2: Loading images with ImageOCRReader (GPT-5 vision) ---")
    reader = ImageOCRReader(model="gpt-5", lang="auto")
    documents = reader.load_data(image_files)

    print(f"Successfully loaded {len(documents)} documents from images.")
    for doc in documents:
        print("\n--- Document ---")
        print(f"Text: {doc.text[:120]}...")
        print(f"Metadata: {doc.metadata}")

    # --- 3. Configure the LlamaIndex environment (GPT-5 + OpenAI embeddings) ---
    print("\n--- Step 3: Setting up LlamaIndex environment ---")
    setup_environment()

    # --- 4. Build the index and run queries ---
    print("\n--- Step 4: Building index and querying ---")
    from llama_index.core import VectorStoreIndex

    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    for q in [
        "What is LlamaIndex?",
        "截图里的用户名是什么？",
        "红色的牌子上写了什么？",
    ]:
        print(f"\nQuerying: {q}")
        print(f"Response: {query_engine.query(q)}")


if __name__ == "__main__":
    main()
