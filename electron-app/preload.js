const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('aiBridge', {
  transformText: async ({ noteContent, selection, prompt }) => {
    const response = await ipcRenderer.invoke('ai:transform', {
      noteContent,
      selection,
      prompt
    });
    return response;
  }
});

contextBridge.exposeInMainWorld('appBridge', {
  version: () => process.versions.electron
});
