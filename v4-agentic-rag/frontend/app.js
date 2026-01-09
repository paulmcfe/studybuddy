// DOM Elements
const messagesContainer = document.getElementById('messagesContainer');
const messagesDiv = document.getElementById('messages');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const versionTag = document.querySelector('.version-tag');

// API Configuration
const API_URL = '/api/chat';
const STATUS_URL = '/api/status';

// State
let isLoading = false;
let indexingComplete = false;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    messageInput.focus();
    setupEventListeners();
    autoResizeTextarea();
    checkIndexingStatus();
});

// Check indexing status
async function checkIndexingStatus() {
    try {
        const response = await fetch(STATUS_URL);
        const status = await response.json();

        if (status.indexing_complete) {
            versionTag.textContent = `StudyBuddy v4 · ${status.documents_indexed} guides indexed (${status.chunks_in_db} chunks)`;
            indexingComplete = true;
        } else {
            const currentFile = status.current_file ? ` "${status.current_file}"` : '';
            versionTag.textContent = `StudyBuddy v4 · Indexing${currentFile}... (${status.chunks_in_db} chunks)`;
            setTimeout(checkIndexingStatus, 500);
        }
    } catch (error) {
        versionTag.textContent = `StudyBuddy v4 · Connecting...`;
        setTimeout(checkIndexingStatus, 2000);
    }
}

// Event Listeners
function setupEventListeners() {
    sendButton.addEventListener('click', handleSend);
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });
    messageInput.addEventListener('input', autoResizeTextarea);
}

function autoResizeTextarea() {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 200) + 'px';
}

// Handle send message
async function handleSend() {
    const message = messageInput.value.trim();
    if (!message || isLoading) return;

    messageInput.value = '';
    messageInput.style.height = 'auto';
    addMessage(message, 'user');
    setLoadingState(true);

    const loadingDiv = addMessage('', 'assistant', true);

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message }),
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || `Server error: ${response.status}`);
        }

        // Render main response with markdown
        loadingDiv.innerHTML = marked.parse(data.reply);

        // Add confidence indicator
        if (data.confidence !== undefined) {
            const confidenceDiv = createConfidenceIndicator(data.confidence);
            loadingDiv.parentElement.appendChild(confidenceDiv);
        }

        // Add flashcard if present
        if (data.flashcard) {
            const flashcardDiv = createFlashcard(data.flashcard);
            loadingDiv.parentElement.appendChild(flashcardDiv);
        }

        // Add analysis toggle
        if (data.analysis) {
            const analysisDiv = createAnalysisToggle(data.analysis);
            loadingDiv.parentElement.appendChild(analysisDiv);
        }

    } catch (error) {
        console.error('Error:', error);
        loadingDiv.parentElement.remove();
        showError(getErrorMessage(error));
    } finally {
        setLoadingState(false);
    }
}

// Create confidence indicator
function createConfidenceIndicator(confidence) {
    const container = document.createElement('div');
    container.className = 'confidence-indicator';

    const percentage = Math.round(confidence * 100);
    const level = confidence >= 0.8 ? 'high' : confidence >= 0.5 ? 'medium' : 'low';

    container.innerHTML = `
        <span class="confidence-label">Confidence:</span>
        <div class="confidence-bar">
            <div class="confidence-fill ${level}" style="width: ${percentage}%"></div>
        </div>
        <span class="confidence-value">${percentage}%</span>
    `;

    return container;
}

// Create flashcard component
function createFlashcard(flashcard) {
    const container = document.createElement('div');
    container.className = 'flashcard-container';

    container.innerHTML = `
        <div class="flashcard-header">
            <span class="flashcard-icon">&#128221;</span>
            <span>Flashcard Suggestion</span>
        </div>
        <div class="flashcard" onclick="this.classList.toggle('flipped')">
            <div class="flashcard-inner">
                <div class="flashcard-front">
                    <div class="flashcard-label">Question</div>
                    <div class="flashcard-text">${escapeHtml(flashcard.front)}</div>
                    <div class="flashcard-hint">Click to reveal answer</div>
                </div>
                <div class="flashcard-back">
                    <div class="flashcard-label">Answer</div>
                    <div class="flashcard-text">${escapeHtml(flashcard.back)}</div>
                    <div class="flashcard-hint">Click to see question</div>
                </div>
            </div>
        </div>
    `;

    // Match heights after render
    requestAnimationFrame(() => {
        const front = container.querySelector('.flashcard-front');
        const back = container.querySelector('.flashcard-back');
        const maxHeight = Math.max(front.offsetHeight, back.scrollHeight);
        front.style.minHeight = maxHeight + 'px';
        back.style.minHeight = maxHeight + 'px';
    });

    return container;
}

// Create analysis toggle
function createAnalysisToggle(analysis) {
    const container = document.createElement('div');
    container.className = 'analysis-container';

    const analysisText = `Query Type: ${analysis.query_type || 'N/A'}
Complexity: ${analysis.complexity || 'N/A'}
Used Retrieval: ${analysis.used_retrieval ? 'Yes' : 'No'}
Documents Retrieved: ${analysis.docs_retrieved || 0}
Retrieval Iterations: ${analysis.iterations || 0}`;

    container.innerHTML = `
        <button class="analysis-toggle">Show analysis</button>
        <div class="analysis-content hidden">
            <pre>${escapeHtml(analysisText)}</pre>
        </div>
    `;

    const toggle = container.querySelector('.analysis-toggle');
    const content = container.querySelector('.analysis-content');
    toggle.addEventListener('click', () => {
        content.classList.toggle('hidden');
        toggle.textContent = content.classList.contains('hidden')
            ? 'Show analysis'
            : 'Hide analysis';
    });

    return container;
}

// Add message to chat
function addMessage(content, role, isLoading = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    if (isLoading) {
        contentDiv.innerHTML = '<div class="loading-dots"><span></span><span></span><span></span></div>';
    } else {
        contentDiv.textContent = content;
    }

    messageDiv.appendChild(contentDiv);
    messagesDiv.appendChild(messageDiv);
    scrollToBottom();

    return contentDiv;
}

// Show error message
function showError(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message error';
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = message;
    messageDiv.appendChild(contentDiv);
    messagesDiv.appendChild(messageDiv);
    scrollToBottom();
}

// Get user-friendly error message
function getErrorMessage(error) {
    if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
        return 'Unable to connect to the server. Please make sure the backend is running.';
    }
    return `Error: ${error.message}`;
}

// Set loading state
function setLoadingState(loading) {
    isLoading = loading;
    sendButton.disabled = loading;
    messageInput.disabled = loading;
    if (!loading) messageInput.focus();
}

// Scroll to bottom of messages
function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
