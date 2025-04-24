// backend/server.js
require('dotenv').config();
const express = require('express');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const multer = require('multer');

const app = express();
const PORT = 8000;  // Port for the Express server

// Middleware for JSON and static file serving
app.use(express.json());
app.use(express.static(path.join(__dirname, '..', 'frontend')));

// Directories for models, datasets, and chat history
const MODELS_DIR = path.join(__dirname, 'models');
const DATASETS_DIR = path.join(__dirname, 'datasets');
const HISTORY_DIR = path.join(__dirname, 'history');

// Ensure necessary directories exist
if (!fs.existsSync(HISTORY_DIR)) {
    fs.mkdirSync(HISTORY_DIR, { recursive: true });
}
if (!fs.existsSync(DATASETS_DIR)) {
    fs.mkdirSync(DATASETS_DIR, { recursive: true });
}

// Inference worker process (Python) and state
let inferenceProcess = null;
let currentModel = null;
let inferenceBusy = false;

// SSE clients for training progress
let trainingClients = [];
let trainingProc = null;  // Training process handle

// Function to spawn the Python inference worker for the given model
function spawnInferenceWorker(modelName) {
    // Kill existing worker if running
    if (inferenceProcess) {
        inferenceProcess.kill();
        inferenceProcess = null;
    }
    // Launch new worker
    const modelPath = path.join(MODELS_DIR, modelName);
    inferenceProcess = spawn('python', ['inference_worker.py', modelPath], {
        cwd: __dirname,
        stdio: ['pipe', 'pipe', 'pipe']
    });
    currentModel = modelName;
    // Log any errors from worker
    inferenceProcess.stderr.on('data', (data) => {
        console.error(`Inference worker [${modelName}] stderr: ${data}`);
    });
    // If worker exits unexpectedly, reset state
    inferenceProcess.on('exit', (code, signal) => {
        console.error(`Inference worker [${modelName}] exited (code=${code}, signal=${signal})`);
        inferenceProcess = null;
        currentModel = null;
        inferenceBusy = false;
    });
}

// GET /models - list available models (folders in models/)
app.get('/models', (req, res) => {
    let models = [];
    try {
        const items = fs.readdirSync(MODELS_DIR, { withFileTypes: true });
        models = items.filter(item => item.isDirectory()).map(item => item.name);
    } catch (err) {
        console.error('Error reading models directory:', err);
        return res.status(500).json({ error: 'Failed to list models' });
    }
    return res.json({ models });
});

// GET /history?model=<name> - get chat history for the specified model
app.get('/history', (req, res) => {
    const modelName = req.query.model;
    if (!modelName) {
        return res.status(400).json({ error: 'Missing model parameter' });
    }
    const safeName = modelName.replace(/[^a-zA-Z0-9_\-]/g, '_');
    const historyFile = path.join(HISTORY_DIR, `${safeName}_history.json`);
    if (!fs.existsSync(historyFile)) {
        return res.json({ history: [] });
    }
    try {
        const data = fs.readFileSync(historyFile, 'utf8');
        const history = JSON.parse(data);
        return res.json({ history });
    } catch (err) {
        console.error('Failed to read history file:', err);
        return res.status(500).json({ error: 'Could not read history' });
    }
});

// POST /chat - send a user message to the model and get a response
app.post('/chat', (req, res) => {
    const modelName = req.body.model;
    const userMessage = req.body.message;
    if (!modelName || !userMessage) {
        return res.status(400).json({ error: 'Model and message are required' });
    }
    // If needed, spawn or switch the inference worker to the selected model
    if (!inferenceProcess || currentModel !== modelName) {
        spawnInferenceWorker(modelName);
    }
    if (!inferenceProcess) {
        return res.status(500).json({ error: 'Failed to start inference worker' });
    }
    // Only one inference at a time (simple queue)
    if (inferenceBusy) {
        return res.status(429).json({ error: 'Inference in progress. Please wait.' });
    }
    inferenceBusy = true;
    // Load existing history for context
    const safeName = modelName.replace(/[^a-zA-Z0-9_\-]/g, '_');
    const historyFile = path.join(HISTORY_DIR, `${safeName}_history.json`);
    let history = [];
    if (fs.existsSync(historyFile)) {
        try {
            const historyData = fs.readFileSync(historyFile, 'utf8');
            history = JSON.parse(historyData);
        } catch (err) {
            console.warn('History parse failed, starting new conversation.');
            history = [];
        }
    }
    // Append the user's message to history
    history.push({ role: 'user', message: userMessage });
    // Build the prompt text from conversation history
    let prompt = '';
    for (const entry of history) {
        if (entry.role === 'user') {
            prompt += `User: ${entry.message}\n`;
        } else if (entry.role === 'assistant') {
            prompt += `Assistant: ${entry.message}\n`;
        }
    }
    prompt += 'Assistant:';  // the model should continue from the assistant role
    // Send prompt to inference worker via stdin
    const payload = JSON.stringify({ type: 'infer', prompt });
    try {
        inferenceProcess.stdin.write(payload + '\n');
    } catch (err) {
        console.error('Error writing to inference worker stdin:', err);
        inferenceBusy = false;
        return res.status(500).json({ error: 'Failed to send prompt to model' });
    }
    // Wait for the model's response
    let buffer = '';
    const onData = (chunk) => {
        buffer += chunk;
        const newlineIndex = buffer.indexOf('\n');
        if (newlineIndex === -1) return; // not a complete response yet
        // Extract one full JSON line
        const jsonLine = buffer.slice(0, newlineIndex);
        buffer = buffer.slice(newlineIndex + 1);
        // Stop listening for more data for this request
        inferenceProcess.stdout.off('data', onData);
        let reply;
        try {
            const parsed = JSON.parse(jsonLine);
            if (parsed.error) {
                inferenceBusy = false;
                return res.status(500).json({ error: parsed.error });
            }
            reply = parsed.response;
        } catch (err) {
            console.error('Failed to parse model response:', jsonLine);
            inferenceBusy = false;
            return res.status(500).json({ error: 'Model response parse error' });
        }
        // Append the assistant's reply to history and save it
        history.push({ role: 'assistant', message: reply });
        try {
            fs.writeFileSync(historyFile, JSON.stringify(history, null, 2));
        } catch (err) {
            console.error('Failed to save history:', err);
        }
        inferenceBusy = false;
        // Send back the assistant's reply
        return res.json({ reply });
    };
    // Attach the data listener for this response
    inferenceProcess.stdout.on('data', onData);
});

// Configure Multer storage for file uploads (datasets)
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, DATASETS_DIR);
    },
    filename: (req, file, cb) => {
        const baseName = req.body.new_model || 'dataset';
        const safeBase = baseName.replace(/[^a-zA-Z0-9_\-]/g, '_');
        const timestamp = Date.now();
        const ext = path.extname(file.originalname) || '.json';
        cb(null, `${safeBase}_${timestamp}${ext}`);
    }
});
const upload = multer({ storage });

// POST /train - accept a dataset file and start fine-tuning
app.post('/train', upload.single('dataset'), (req, res) => {
    const baseModel = req.body.base_model;
    const newModel = req.body.new_model;
    if (!baseModel || !newModel || !req.file) {
        return res.status(400).json({ error: 'base_model, new_model, and dataset file are required' });
    }
    if (trainingProc) {
        return res.status(429).json({ error: 'A training job is already in progress' });
    }
    // Paths for base model, output model, and dataset
    const baseModelPath = path.join(MODELS_DIR, baseModel);
    const outputModelPath = path.join(MODELS_DIR, newModel);
    const dataFilePath = req.file.path;
    try {
        trainingProc = spawn('python', [
            'train.py',
            '--base_model', baseModelPath,
            '--data_file', dataFilePath,
            '--output_dir', outputModelPath
        ], {
            cwd: __dirname,
            stdio: ['ignore', 'pipe', 'pipe']
        });
    } catch (err) {
        console.error('Failed to start training process:', err);
        return res.status(500).json({ error: 'Failed to start training process' });
    }
    // Forward training output to SSE clients
    trainingProc.stdout.on('data', (chunk) => {
        const message = chunk.toString();
        trainingClients.forEach(clientRes => {
            clientRes.write(`data: ${message.replace(/\r?\n$/, '')}\n\n`);
        });
    });
    trainingProc.stderr.on('data', (chunk) => {
        const message = chunk.toString();
        trainingClients.forEach(clientRes => {
            clientRes.write(`data: ${message.replace(/\r?\n$/, '')}\n\n`);
        });
    });
    // On training completion, notify clients and reset
    trainingProc.on('exit', (code, signal) => {
        const completion = (code === 0)
            ? 'Training completed.'
            : `Training process exited with code ${code}.`;
        trainingClients.forEach(clientRes => {
            clientRes.write(`event: done\ndata: ${completion}\n\n`);
            clientRes.end();
        });
        trainingClients = [];
        trainingProc = null;
    });
    // Respond immediately that training has started
    return res.json({ status: 'started' });
});

// GET /train/stream - SSE endpoint for training progress
app.get('/train/stream', (req, res) => {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();
    trainingClients.push(res);
    req.on('close', () => {
        trainingClients = trainingClients.filter(clientRes => clientRes !== res);
    });
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running at http://localhost:${PORT}`);
});
