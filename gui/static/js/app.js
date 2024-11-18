let currentEditId = null;
const API_BASE = '/api';
let userList = [];

async function loadUserList() {
    try {
        const response = await fetch(`${API_BASE}/users`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        userList = await response.json();
        populateUserDropdowns();
    } catch (error) {
        console.error('Error loading users:', error);
        showDebug('Error loading user list: ' + error.message);
    }
}

async function validateDatabase() {
    try {
        const response = await fetch(`${API_BASE}/validate`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const stats = await response.json();
        showDebug(`Database Validation:
            Total Memories: ${stats.total_memories}
            Active Memories: ${stats.active_memories}
            Total Users: ${stats.total_users}
            Index Size: ${stats.index_size}
            Issues Found: ${Object.values(stats.validation).flat().length}`);
        return stats;
    } catch (error) {
        console.error('Error validating database:', error);
        showDebug('Error validating database: ' + error.message);
    }
}

function populateUserDropdowns() {
    const searchUserSelect = document.getElementById('userId');
    const newMemoryUserSelect = document.getElementById('newMemoryUserId');
    
    // Clear existing options
    searchUserSelect.innerHTML = '<option value="">All Users</option>';
    newMemoryUserSelect.innerHTML = '<option value="">Select User</option>';
    
    // Add user options
    userList.forEach(userId => {
        searchUserSelect.add(new Option(userId, userId));
        newMemoryUserSelect.add(new Option(userId, userId));
    });
}

// Search memories
async function searchMemories() {
    const query = document.getElementById('searchQuery').value;
    const userId = document.getElementById('userId').value;
    
    try {
        const response = await fetch(`${API_BASE}/memories/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query || "test",  // Default query if empty
                user_id: userId || null,
                limit: 100
            }),
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const memories = await response.json();
        console.log('Search results:', memories);  // Debug log
        displayResults(memories);
    } catch (error) {
        console.error('Error:', error);
        alert('Error searching memories: ' + error.message);
    }
}

// Create new memory
async function createMemory() {
    const userId = document.getElementById('newMemoryUserId').value;
    const content = document.getElementById('newMemoryContent').value;
    
    if (!userId || !content) {
        alert('Please fill in both User ID and content');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/memories`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_id: userId,
                content: content
            }),
        });
        
        if (response.ok) {
            document.getElementById('newMemoryContent').value = '';
            alert('Memory created successfully');
            // Refresh search results if there's a current search
            if (document.getElementById('searchQuery').value) {
                searchMemories();
            }
        } else {
            alert('Error creating memory');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error creating memory');
    }
}

// Display search results
function displayResults(memories) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';
    
    if (!memories || memories.length === 0) {
        resultsDiv.innerHTML = '<p>> No neural fragments found in database</p>';
        return;
    }
    
    memories.forEach(memory => {
        const memoryCard = document.createElement('div');
        memoryCard.className = 'memory-card';
        
        const relevanceScore = memory.relevance_score !== null && memory.relevance_score !== undefined
            ? `<span class="relevance-score">Neural Match: ${memory.relevance_score.toFixed(3)}</span>` 
            : '';
        
        // Escape content for safe HTML insertion
        const content = memory.content ? memory.content.replace(/</g, '&lt;').replace(/>/g, '&gt;') : 'No content';
        const memoryId = memory.memory_id !== undefined ? memory.memory_id : 'N/A';
        
        memoryCard.innerHTML = `
            <div class="memory-content">${content}</div>
            <div class="memory-metadata">
                ID: ${memoryId}
                ${relevanceScore}
            </div>
            <div class="btn-group">
                <button onclick="editMemory('${memoryId}', this.closest('.memory-card').querySelector('.memory-content').textContent)" 
                        class="btn btn-primary">Reconstruct</button>
                <button onclick="deleteMemory(${memoryId})" 
                        class="btn btn-danger">Purge</button>
            </div>
        `;
        
        resultsDiv.appendChild(memoryCard);
    });
}

// Run initial search on page load
document.addEventListener('DOMContentLoaded', async () => {
    console.log('Page loaded, initializing...');
    showDebug('Validating database...');
    
    // Validate database first
    const stats = await validateDatabase();
    if (stats) {
        showDebug(`Database validated successfully.
            Active memories: ${stats.active_memories}
            Total users: ${stats.total_users}`);
    }
    
    // Load user list
    await loadUserList();
    
    // Run initial search
    searchMemories();
});

// Edit memory
function editMemory(id, content) {
    currentEditId = id;
    // Remove any leading/trailing whitespace and set the content
    document.getElementById('editMemoryContent').value = content.trim();
    new bootstrap.Modal(document.getElementById('editModal')).show();
}

// Save edited memory
async function saveEdit() {
    if (!currentEditId) {
        console.error('No memory ID set for editing');
        return;
    }
    
    const content = document.getElementById('editMemoryContent').value;
    
    if (!content.trim()) {
        alert('Content cannot be empty');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/memories/${currentEditId}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                content: content.trim()
            }),
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Error updating memory');
        }
        
        // Close modal and refresh results
        const modal = bootstrap.Modal.getInstance(document.getElementById('editModal'));
        modal.hide();
        await searchMemories();
        showDebug('Memory successfully reconstructed');
        
        // Reset currentEditId
        currentEditId = null;
    } catch (error) {
        console.error('Error:', error);
        alert('Error updating memory: ' + error.message);
    }
}

// Delete memory
async function deleteMemory(id) {
    if (!confirm('Are you sure you want to delete this memory?')) return;
    
    try {
        const response = await fetch(`${API_BASE}/memories/${id}`, {
            method: 'DELETE',
        });
        
        if (response.ok) {
            await searchMemories(); // Refresh the view after deletion
            showDebug('Memory deleted successfully');
        } else {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Error deleting memory');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error deleting memory: ' + error.message);
    }
} 