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

// Check server status
async function checkIndexingStatus() {
    try {
        await fetch(STATUS_URL);
        versionTag.textContent = `StudyBuddy v2 · RAG enabled`;
        indexingComplete = true;
    } catch (error) {
        // Server not ready yet, try again
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

// Auto-resize textarea
function autoResizeTextarea() {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 200) + 'px';
}

// Handle send message
async function handleSend() {
    const message = messageInput.value.trim();

    if (!message || isLoading) return;

    // Clear input and reset height
    messageInput.value = '';
    messageInput.style.height = 'auto';

    // Add user message to chat
    addMessage(message, 'user');

    // Disable input while loading
    setLoadingState(true);

    // Show loading indicator
    const loadingDiv = addMessage('', 'assistant', true);

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message }),
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || `Server error: ${response.status}`);
        }

        // Replace loading indicator with actual response (render markdown)
        loadingDiv.innerHTML = marked.parse(data.reply);

    } catch (error) {
        console.error('Error:', error);
        loadingDiv.parentElement.remove();
        showError(getErrorMessage(error));
    } finally {
        setLoadingState(false);
    }
}

// Add a message to the chat
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

// Set loading state (enable/disable input)
function setLoadingState(loading) {
    isLoading = loading;
    sendButton.disabled = loading;
    messageInput.disabled = loading;

    if (!loading) {
        messageInput.focus();
    }
}

// Scroll to bottom of messages
function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}
