// frontend/script.js

// Utility: escape HTML to prevent injection in chat messages (basic)
function escapeHTML(str) {
    return str.replace(/&/g, "&amp;")
              .replace(/</g, "&lt;")
              .replace(/>/g, "&gt;")
              .replace(/"/g, "&quot;")
              .replace(/'/g, "&#39;");
}

const modelSelect = document.getElementById('modelSelect');
const baseModelSelect = document.getElementById('baseModelSelect');
const newModelNameInput = document.getElementById('newModelName');
const datasetFileInput = document.getElementById('datasetFile');
const trainForm = document.getElementById('trainForm');
const trainButton = document.getElementById('trainButton');
const trainOutput = document.getElementById('trainOutput');
const chatArea = document.getElementById('chatArea');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');

// Load available models from backend and populate selects
async function loadModels() {
    try {
        const res = await fetch('/models');
        const data = await res.json();
        const models = data.models || [];
        // Clear existing options
        modelSelect.innerHTML = '';
        baseModelSelect.innerHTML = '';
        models.forEach(modelName => {
            const opt1 = document.createElement('option');
            opt1.value = modelName;
            opt1.textContent = modelName;
            modelSelect.appendChild(opt1);
            // Base model select might include all models (assuming user knows which are base)
            const opt2 = document.createElement('option');
            opt2.value = modelName;
            opt2.textContent = modelName;
            baseModelSelect.appendChild(opt2);
        });
        // Select first model by default in both
        if (models.length > 0) {
            modelSelect.value = models[0];
            baseModelSelect.value = models[0];
        }
    } catch (err) {
        console.error('Failed to load models:', err);
    }
}

// Display chat history for a given model
async function loadHistory(modelName) {
    try {
        const res = await fetch(`/history?model=${encodeURIComponent(modelName)}`);
        const data = await res.json();
        const history = data.history || [];
        chatArea.innerHTML = '';  // clear current chat display
        history.forEach(entry => {
            const msgDiv = document.createElement('div');
            msgDiv.classList.add('message');
            msgDiv.classList.add(entry.role === 'user' ? 'user' : 'assistant');
            // Prefix with role name and escape content
            msgDiv.textContent = (entry.role === 'user' ? 'User: ' : 'Assistant: ') + entry.message;
            chatArea.appendChild(msgDiv);
        });
        // Scroll to bottom of chat area
        chatArea.scrollTop = chatArea.scrollHeight;
    } catch (err) {
        console.error('Failed to load history:', err);
    }
}

// When user changes the selected model in chat, load its history
modelSelect.addEventListener('change', () => {
    const model = modelSelect.value;
    loadHistory(model);
});

// Handle training form submission
trainForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const baseModel = baseModelSelect.value;
    const newModel = newModelNameInput.value.trim();
    const file = datasetFileInput.files[0];
    if (!baseModel || !newModel || !file) {
        alert("Please select a base model, provide a new model name, and choose a dataset file.");
        return;
    }
    // Disable the train button to prevent multiple submissions
    trainButton.disabled = true;
    trainOutput.textContent = "";  // clear previous logs
    // Open an EventSource for training progress
    const evtSource = new EventSource('/train/stream');
    // Append logs to trainOutput as they come
    evtSource.onmessage = (event) => {
        trainOutput.textContent += event.data + "\n";
        trainOutput.scrollTop = trainOutput.scrollHeight;
    };
    // When training is done, re-enable form and update model list
    evtSource.addEventListener('done', (event) => {
        const message = event.data;
        trainOutput.textContent += `\n${message}\n`;
        // Close the SSE connection
        evtSource.close();
        // Re-enable the train button for another round if needed
        trainButton.disabled = false;
        // Refresh model list to include the new model and select it in chat
        loadModels().then(() => {
            modelSelect.value = newModel;
            loadHistory(newModel);
        });
    });
    evtSource.onerror = (err) => {
        console.error("SSE error:", err);
        evtSource.close();
        trainButton.disabled = false;
    };
    // Once SSE connection is open, send the training request
    evtSource.onopen = () => {
        const formData = new FormData();
        formData.append('base_model', baseModel);
        formData.append('new_model', newModel);
        formData.append('dataset', file);
        fetch('/train', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                // Display error in training output
                trainOutput.textContent += `Error: ${data.error}\n`;
                // Close SSE and re-enable form on failure
                evtSource.close();
                trainButton.disabled = false;
            } else {
                console.log("Training job started");
            }
        })
        .catch(err => {
            console.error("Training request failed:", err);
            trainOutput.textContent += `Error: Failed to start training.\n`;
            evtSource.close();
            trainButton.disabled = false;
        });
    };
});

// Handle sending a chat message
sendButton.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        sendMessage();
    }
});
function sendMessage() {
    const message = userInput.value.trim();
    if (message === "") return;
    const model = modelSelect.value;
    // Append the user's message to the chat area
    const userDiv = document.createElement('div');
    userDiv.classList.add('message', 'user');
    userDiv.textContent = "User: " + message;
    chatArea.appendChild(userDiv);
    chatArea.scrollTop = chatArea.scrollHeight;
    // Clear input and disable controls while waiting
    userInput.value = "";
    userInput.disabled = true;
    sendButton.disabled = true;
    // Send the message to the server
    fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: model, message: message })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            // Display error as assistant message
            const errDiv = document.createElement('div');
            errDiv.classList.add('message', 'assistant');
            errDiv.textContent = "Error: " + data.error;
            chatArea.appendChild(errDiv);
        } else {
            const reply = data.reply;
            const botDiv = document.createElement('div');
            botDiv.classList.add('message', 'assistant');
            botDiv.textContent = "Assistant: " + reply;
            chatArea.appendChild(botDiv);
        }
        chatArea.scrollTop = chatArea.scrollHeight;
    })
    .catch(err => {
        console.error("Error sending message:", err);
        const errDiv = document.createElement('div');
        errDiv.classList.add('message', 'assistant');
        errDiv.textContent = "Error: Failed to get response.";
        chatArea.appendChild(errDiv);
    })
    .finally(() => {
        // Re-enable input and button
        userInput.disabled = false;
        sendButton.disabled = false;
        userInput.focus();
    });
}

// Initialize the page by loading models and history for default model
window.addEventListener('DOMContentLoaded', () => {
    loadModels().then(() => {
        if (modelSelect.value) {
            loadHistory(modelSelect.value);
        }
    });
});
