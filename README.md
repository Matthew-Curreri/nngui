# LLaMA 3.3 Local Training and Chat System

This project provides a local system for fine-tuning and chatting with **LLaMA 3.3** models. It consists of a backend (Node.js + Python) and a frontend (vanilla JavaScript) that allow you to:

- **Fine-tune a LLaMA 3.3 model** on custom instruction data (using Low-Rank Adaptation, LoRA).
- **Chat with the model** in an interactive interface, with conversation history persistence.

## Project Structure

llama-trainer/ ├── backend/ │ ├── server.js # Node.js Express server │ ├── inference_worker.py # Python worker for model inference │ ├── train.py # Python script to fine-tune (train) the model │ ├── models/ # Directory for model files (base and fine-tuned) │ ├── datasets/ # Directory for training datasets │ └── history/ # Directory where chat history logs are stored ├── frontend/ │ ├── index.html # Frontend UI (single-page application) │ ├── style.css # Styles for the UI │ └── script.js # Client-side JS for interacting with backend ├── README.md # (This file) Project overview and instructions

A sample Alpaca-style dataset (`datasets/sample_dataset.json`) is provided for testing, and a placeholder model directory (`backend/models/llama-3.3-7B`) is included with instructions.

## Requirements

- **Hardware:** A Linux machine with an NVIDIA 4090 GPU (24GB VRAM) and an Intel i9-13900K CPU (or similar). The system uses 4-bit quantization and LoRA to fit fine-tuning in GPU memory&#8203;:contentReference[oaicite:6]{index=6}.
- **Software:** 
  - Python 3.8+ with packages: `torch`, `transformers`, `datasets`, `peft`, `bitsandbytes`, `accelerate`. (Install with `pip`, e.g. `pip install transformers accelerate peft bitsandbytes datasets`.)
  - Node.js 18+ with `express` and `multer` (`npm install express multer`).
  - LLaMA 3.3 model weights (e.g., 7B) converted to Hugging Face format. *You must have accepted Meta's license to obtain these.* Place the model files in `backend/models/llama-3.3-7B` (see that folder's README).

## Setup Instructions

1. **Model Files:** After downloading/converting the LLaMA 3.3 model, create a directory under `backend/models/` (for example, `llama-3.3-7B`) and put the model files there (including `config.json`, `tokenizer.model`, and `pytorch_model.bin` or similar). Refer to `backend/models/llama-3.3-7B/README.md` for details.
2. **Install Dependencies:** 
   - Install Python libraries: `pip install torch transformers accelerate peft bitsandbytes datasets`.
   - Install Node.js libraries: `npm install express multer` (run this in the `backend` directory or adjust import paths as needed).
3. **Start the Backend Server:** From the `backend/` directory, run `node server.js`. This will:
   - Launch the Express server (by default at `http://localhost:8000`).
   - Spawn a Python `inference_worker.py` process on-demand for model inference.
   - Expose APIs for listing models, starting training, and chat inference.
4. **Open the Frontend:** In a web browser, open `http://localhost:8000`. This will load the `frontend/index.html` page (served by the Express server).

## Using the System

### Fine-tuning a Model

1. **Select Base Model:** In the "Train a Model" section of the UI, choose a base model from the dropdown (populated from the `models/` directory).
2. **Name the New Model:** Enter a name for your fine-tuned model. A new directory with this name will be created under `backend/models/` to save the LoRA adapter.
3. **Upload Training Data:** Choose a JSON file with your instruction-following dataset. The dataset should be in the Alpaca format&#8203;:contentReference[oaicite:7]{index=7}, i.e., a list of records with `"instruction"`, `"input"`, and `"output"` fields. (See `datasets/sample_dataset.json` for an example.)
4. **Start Training:** Click "Start Training". The backend will launch the `train.py` script to fine-tune the model. Training progress is streamed to the UI in real-time using Server-Sent Events (SSE).
   - The training script uses 4-bit precision and LoRA for efficiency&#8203;:contentReference[oaicite:8]{index=8}. Only a small percentage of model parameters (the LoRA adapters) are trainable&#8203;:contentReference[oaicite:9]{index=9}, significantly reducing GPU memory usage.
   - The script formats each example with the same prompt template used by Stanford Alpaca&#8203;:contentReference[oaicite:10]{index=10}&#8203;:contentReference[oaicite:11]{index=11} and fine-tunes the model for a few epochs (default 3). It uses a learning rate of 3e-4 and LoRA rank 16 targeting the model's key/query/value projection layers&#8203;:contentReference[oaicite:12]{index=12}.
5. **Completion:** When training finishes, a new subfolder will appear in `backend/models/` with the given new model name, containing the LoRA adapter weights (and a copy of the tokenizer). The UI will automatically refresh the model list and select the new model for chatting.

### Chatting with the Model

1. **Select Model for Chat:** In the "Chat with Model" section, choose a model from the dropdown. You can select either a base model or a fine-tuned model (e.g., the one you just trained).
2. **Conversation History:** Upon selection, the UI will load any saved conversation history for that model. History is persisted in `backend/history/<model>_history.json` and will include all prior user and assistant messages.
3. **Send Messages:** Type a message in the input box and click "Send". The message will be sent to the backend (`/chat` endpoint), which will use the Python inference worker to generate a response from the model.
   - The inference worker loads the model (with LoRA applied, if applicable) in 4-bit mode for fast GPU inference&#8203;:contentReference[oaicite:13]{index=13}. It constructs a prompt from the conversation history and the new user query, then generates the assistant's answer.
   - The assistant's reply will appear in the chat area. The conversation (both user query and assistant answer) is saved to the history file and will persist if you switch models or refresh the page.
4. **Continue the Conversation:** You can send follow-up questions or prompts, and the model will respond in context, considering the history of the conversation. (Keep in mind the model has a limited context window — by default around 2048 tokens for LLaMA.)

### Live Monitoring and Logs

- Training logs (loss updates, etc.) are shown in the "Train a Model" section as the training progresses. These logs come directly from the `train.py` script via SSE.
- If any error occurs (e.g., out-of-memory or missing files), it will be displayed in the UI (and logged in the terminal). Ensure your model files and dataset are correctly prepared.

## Sample Dataset

A sample dataset file `datasets/sample_dataset.json` is included to demonstrate the expected format. It contains a few simple instruction-output examples (e.g., writing a short poem, translating a phrase, etc.). You can use this sample to test fine-tuning, though for meaningful results a larger dataset is recommended.

## Technical Details

- **LoRA Fine-Tuning:** This system uses [Hugging Face PEFT](https://github.com/huggingface/peft) to apply LoRA. Only the LoRA adapter weights are saved (the base model remains unchanged)&#8203;:contentReference[oaicite:14]{index=14}. The `train.py` script prepares the Alpaca-style prompt for each training example and updates the model accordingly. Hyperparameters (e.g., learning rate, epochs) can be adjusted via command-line arguments in `train.py`.
- **4-bit Quantization:** Both training and inference leverage 4-bit quantization (via the [bitsandbytes library](https://github.com/TimDettmers/bitsandbytes)) to reduce memory usage&#8203;:contentReference[oaicite:15]{index=15}. This allows fine-tuning a 7B parameter model on a 24GB GPU and speeds up inference.
- **Inference Worker:** The Python `inference_worker.py` stays running to serve inference requests. It loads the model (and merges LoRA weights on the fly&#8203;:contentReference[oaicite:16]{index=16}) and uses `model.generate()` to produce responses. The worker communicates with the Node server via JSON over stdin/stdout.
- **Express Server:** The Node `server.js` orchestrates everything. It provides endpoints to list models, initiate training (spawning the Python process and streaming its output), and handle chat requests. It also serves the frontend files. Chat history is stored on the server after each interaction, enabling persistent conversations.

## References

- Stanford Alpaca project and dataset format&#8203;:contentReference[oaicite:17]{index=17}&#8203;:contentReference[oaicite:18]{index=18} – the instruction tuning approach and data format that this system is based on.
- Hugging Face Transformers and PEFT documentation – for integration of 4-bit quantization&#8203;:contentReference[oaicite:19]{index=19} and LoRA fine-tuning techniques.
- Tloen's Alpaca-LoRA training notes&#8203;:contentReference[oaicite:20]{index=20} – hyperparameter choices (LoRA rank, target modules, learning rate) that informed the defaults used here.
