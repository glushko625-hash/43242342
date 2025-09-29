import { marked } from '../node_modules/marked/lib/marked.esm.js';

const notesList = document.getElementById('notes-list');
const noteTitle = document.getElementById('note-title');
const noteBody = document.getElementById('note-body');
const createNoteButton = document.getElementById('create-note');
const status = document.getElementById('status');
const previewContent = document.getElementById('preview-content');
const runAiButton = document.getElementById('run-ai');
const aiPromptInput = document.getElementById('ai-prompt');
const aiOutputSection = document.getElementById('ai-output');
const aiContent = document.getElementById('ai-content');
const aiProvider = document.getElementById('ai-provider');
const applyAiButton = document.getElementById('apply-ai');
const discardAiButton = document.getElementById('discard-ai');
const electronVersion = document.getElementById('electron-version');

const STORAGE_KEY = 'contextual-notes';
let notes = [];
let activeNoteId = null;
let lastSelectionRange = null;
let pendingAiContent = null;

function generateId() {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }

  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

marked.setOptions({
  breaks: true
});

function loadNotes() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) {
    return [];
  }

  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) {
      return parsed;
    }
  } catch (error) {
    console.error('Failed to parse notes', error);
  }

  return [];
}

function persistNotes() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(notes));
}

function createNote() {
  const now = new Date().toISOString();
  const note = {
    id: generateId(),
    title: 'Новая заметка',
    body: '',
    updatedAt: now
  };

  notes.unshift(note);
  activeNoteId = note.id;
  persistNotes();
  renderNotesList();
  loadActiveNote();
}

function renderNotesList() {
  notesList.innerHTML = '';
  notes
    .sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt))
    .forEach((note) => {
      const item = document.createElement('li');
      item.dataset.id = note.id;
      item.classList.toggle('active', note.id === activeNoteId);
      item.innerHTML = `
        <strong>${note.title || 'Без названия'}</strong>
        <small>${new Date(note.updatedAt).toLocaleString()}</small>
      `;
      item.addEventListener('click', () => {
        if (activeNoteId !== note.id) {
          activeNoteId = note.id;
          renderNotesList();
          loadActiveNote();
        }
      });
      notesList.appendChild(item);
    });
}

function loadActiveNote() {
  const note = notes.find((n) => n.id === activeNoteId);
  if (!note) {
    noteTitle.value = '';
    noteBody.value = '';
    updatePreview('');
    return;
  }

  noteTitle.value = note.title;
  noteBody.value = note.body;
  updatePreview(note.body);
}

function updateActiveNote(partial) {
  const note = notes.find((n) => n.id === activeNoteId);
  if (!note) {
    return;
  }

  Object.assign(note, partial);
  note.updatedAt = new Date().toISOString();
  persistNotes();
  renderNotesList();
}

function updatePreview(content) {
  previewContent.innerHTML = marked.parse(content || '');
}

function withActiveNote(callback) {
  const note = notes.find((n) => n.id === activeNoteId);
  if (!note) {
    status.textContent = 'Создайте новую заметку';
    return;
  }
  callback(note);
}

function showStatus(message, duration = 2000) {
  status.textContent = message;
  if (duration) {
    setTimeout(() => {
      if (status.textContent === message) {
        status.textContent = '';
      }
    }, duration);
  }
}

async function runAiTransform() {
  if (runAiButton.disabled) {
    return;
  }

  const prompt = aiPromptInput.value.trim();
  if (!prompt) {
    showStatus('Введите инструкцию для ИИ');
    aiPromptInput.focus();
    return;
  }

  withActiveNote(async (note) => {
    const selectionStart = noteBody.selectionStart;
    const selectionEnd = noteBody.selectionEnd;

    if (selectionStart === selectionEnd) {
      showStatus('Выделите текст в заметке');
      noteBody.focus();
      return;
    }

    const selection = noteBody.value.slice(selectionStart, selectionEnd);
    runAiButton.disabled = true;
    showStatus('ИИ думает...');

    try {
      const response = await window.aiBridge.transformText({
        noteContent: note.body,
        selection,
        prompt
      });

      if (!response?.success) {
        throw new Error(response?.error || 'Не удалось получить ответ');
      }

      pendingAiContent = response.content;
      lastSelectionRange = { start: selectionStart, end: selectionEnd };
      aiContent.textContent = pendingAiContent;
      aiProvider.textContent = response.metadata?.provider === 'openai'
        ? 'OpenAI'
        : 'Локальный режим';
      aiOutputSection.hidden = false;
      showStatus('Готово');
    } catch (error) {
      console.error(error);
      showStatus('Ошибка при обращении к ИИ');
    } finally {
      runAiButton.disabled = false;
    }
  });
}

function applyAiSuggestion() {
  if (!pendingAiContent || !lastSelectionRange) {
    return;
  }

  withActiveNote((note) => {
    const { start, end } = lastSelectionRange;
    const before = noteBody.value.slice(0, start);
    const after = noteBody.value.slice(end);
    const updatedBody = `${before}${pendingAiContent}${after}`;

    noteBody.value = updatedBody;
    updateActiveNote({ body: updatedBody });
    updatePreview(updatedBody);
    pendingAiContent = null;
    lastSelectionRange = null;
    aiOutputSection.hidden = true;
    showStatus('Изменение применено');
  });
}

function discardAiSuggestion() {
  pendingAiContent = null;
  lastSelectionRange = null;
  aiOutputSection.hidden = true;
}

function bootstrap() {
  notes = loadNotes();
  if (notes.length > 0) {
    activeNoteId = notes[0].id;
  }
  renderNotesList();
  loadActiveNote();
  electronVersion.textContent = window.appBridge.version();
}

noteTitle.addEventListener('input', (event) => {
  const value = event.target.value;
  updateActiveNote({ title: value });
});

noteBody.addEventListener('input', (event) => {
  const value = event.target.value;
  updateActiveNote({ body: value });
  updatePreview(value);
});

createNoteButton.addEventListener('click', () => {
  createNote();
  showStatus('Заметка создана');
});

runAiButton.addEventListener('click', runAiTransform);
applyAiButton.addEventListener('click', applyAiSuggestion);
discardAiButton.addEventListener('click', discardAiSuggestion);

window.addEventListener('DOMContentLoaded', bootstrap);
