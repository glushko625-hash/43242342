const https = require('https');

function streamJson(requestOptions, payload) {
  return new Promise((resolve, reject) => {
    const req = https.request(requestOptions, (res) => {
      let data = '';

      res.on('data', (chunk) => {
        data += chunk;
      });

      res.on('end', () => {
        try {
          const parsed = JSON.parse(data);
          resolve(parsed);
        } catch (error) {
          reject(error);
        }
      });
    });

    req.on('error', reject);

    if (payload) {
      req.write(JSON.stringify(payload));
    }

    req.end();
  });
}

function buildFallbackResponse({ prompt, selection }) {
  if (!selection || selection.trim().length === 0) {
    return 'No text was selected for transformation.';
  }

  const shouldSummarize = /summary|summarize|кратко|итог/i.test(prompt);
  const shouldImprove = /improve|улучш/i.test(prompt);

  if (shouldSummarize) {
    const sentences = selection.split(/(?<=[.!?])\s+/);
    const summary = sentences.slice(0, 2).join(' ');
    return `Summary: ${summary}`;
  }

  if (shouldImprove) {
    return selection.replace(/\bочень\b/gi, 'исключительно').replace(/\bgood\b/gi, 'excellent');
  }

  return `${selection}\n\n[AI suggestion unavailable offline]`;
}

async function handleAiTransform(_event, { noteContent, selection, prompt }) {
  const apiKey = process.env.OPENAI_API_KEY;
  const requestBody = {
    model: 'gpt-3.5-turbo',
    messages: [
      {
        role: 'system',
        content: 'You are an assistant that rewrites highlighted snippets inside a note while respecting the surrounding context.'
      },
      {
        role: 'user',
        content: `Note context:\n${noteContent}\n\nSelected text:\n${selection}\n\nInstruction:\n${prompt}`
      }
    ],
    temperature: 0.7
  };

  if (!apiKey) {
    return {
      success: true,
      content: buildFallbackResponse({ prompt, selection }),
      metadata: {
        provider: 'fallback',
        reason: 'OPENAI_API_KEY not set'
      }
    };
  }

  const requestOptions = {
    hostname: 'api.openai.com',
    path: '/v1/chat/completions',
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    }
  };

  try {
    const response = await streamJson(requestOptions, requestBody);
    const content = response.choices?.[0]?.message?.content?.trim();
    if (!content) {
      throw new Error('No content received from API');
    }

    return {
      success: true,
      content,
      metadata: {
        provider: 'openai'
      }
    };
  } catch (error) {
    return {
      success: true,
      content: buildFallbackResponse({ prompt, selection }),
      metadata: {
        provider: 'fallback',
        reason: error.message
      }
    };
  }
}

module.exports = {
  handleAiTransform
};
