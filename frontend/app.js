const $ = (id) => document.getElementById(id);
let busy = false;
let lastText = '';
let currentSettings = null;
let trainingFile = null;
const number = new Intl.NumberFormat('en-US');

function notify(message, error = false) {
  $('notice').textContent = message;
  $('notice').classList.toggle('error', error);
  $('notice').hidden = false;
}

async function api(path, data) {
  let response;
  try {
    response = await fetch(`/api/v1/${path}`, data === undefined ? {} : {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data),
    });
  } catch {
    throw new Error('Cannot connect to the server. Make sure the application is running and try again.');
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    if (response.status === 503) throw new Error('The model is still loading. Please try again shortly.');
    const detail = body.detail;
    if (typeof detail === 'string') {
      if (detail.startsWith('No training texts found for topic:')) throw new Error('No texts in the corpus contain this topic. Try another topic or leave the field empty.');
      throw new Error(detail);
    }
    if (Array.isArray(detail)) throw new Error(detail.map((item) => item.msg).join('; '));
    throw new Error(`Request failed (${response.status}). Please try again.`);
  }
  return response.json();
}

function updateProgress(progress) {
  $('progress-stage').textContent = progress.stage;
  $('progress-percent').textContent = `${progress.percent}%`;
  $('progress-bar').value = progress.percent;
  $('progress-count').textContent = `${number.format(progress.completed)} / ${number.format(progress.total)}`;
  $('progress-time').textContent = `${progress.elapsed.toFixed(1)} s in this stage`;
}

async function streamTraining(path, data) {
  const panel = $('training-progress');
  panel.hidden = false;
  panel.classList.remove('failed');
  updateProgress({ stage: data instanceof FormData ? 'Uploading file to the server' : 'Waiting for training', percent: 0, completed: 0, total: 0, elapsed: 0 });
  let result = null;
  try {
    const multipart = data instanceof FormData;
    const response = await fetch(`/api/v1/${path}/stream`, {
      method: 'POST', ...(multipart ? {} : { headers: { 'Content-Type': 'application/json' } }),
      body: multipart ? data : JSON.stringify(data),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(typeof body.detail === 'string' ? body.detail : `Request failed (${response.status}). Check your settings.`);
    }
    if (!response.body) throw new Error('Your browser does not support streaming responses.');
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    try {
      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value, { stream: !done });
        let boundary;
        while ((boundary = buffer.indexOf('\n\n')) >= 0) {
          const frame = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          const event = frame.split('\n').find((line) => line.startsWith('event: '))?.slice(7);
          const raw = frame.split('\n').filter((line) => line.startsWith('data: ')).map((line) => line.slice(6)).join('\n');
          if (!raw) continue;
          const payload = JSON.parse(raw);
          if (event === 'progress') updateProgress(payload);
          if (event === 'error') throw new Error(payload.message);
          if (event === 'complete') result = payload;
        }
        if (done) break;
      }
    } finally { await reader.cancel().catch(() => { }); reader.releaseLock(); }
    if (!result) throw new Error('The connection closed before training completed. Refresh the statistics to check the model status.');
    $('progress-stage').textContent = 'Done';
    $('progress-percent').textContent = '100%';
    $('progress-bar').value = 100;
    return result;
  } catch (error) {
    panel.classList.add('failed');
    $('progress-stage').textContent = 'Training not confirmed';
    throw error;
  }
}

function setStats(stats) {
  currentSettings = stats.settings;
  $('tokenizer').value = stats.settings.tokenizer;
  updateTokenizerHint();
  $('n-gramm').value = stats.settings.n_gramm;
  $('min-frequency').value = stats.settings.min_frequency;
  $('max-length').value = stats.settings.max_length;
  $('texts-count').textContent = number.format(stats.texts_count);
  $('vocab-size').textContent = number.format(stats.vocab_size);
  $('contexts-count').textContent = number.format(stats.contexts_count);
  $('status').className = 'status ready';
  $('status').replaceChildren(Object.assign(document.createElement('i')), document.createTextNode('Model ready'));
}

async function refresh(showError = false) {
  $('refresh-button').disabled = true;
  try { setStats(await api('model')); }
  catch (error) {
    $('status').className = 'status offline';
    $('status').replaceChildren(document.createElement('i'), document.createTextNode('Server unavailable'));
    if (showError) notify(error.message, true);
  } finally { $('refresh-button').disabled = false; }
}

function setBusy(value, training = false) {
  busy = value;
  $('generate-button').disabled = value;
  $('train-button').disabled = value;
  $('import-txt-button').disabled = value;
  $('training-file').disabled = value;
  $('remove-file-button').disabled = value;
  $('training-texts').disabled = value;
  $('settings-fields').disabled = value;
  $('settings-button').disabled = value;
  $('refresh-button').disabled = value;
  $('generate-button').firstChild.textContent = value && !training ? 'Generating continuation… ' : 'Continue text ';
  $('train-button').firstChild.textContent = value && training ? 'Training model… ' : 'Train model ';
}

$('prefix').addEventListener('input', () => { $('char-count').textContent = `${number.format($('prefix').value.length)} / 10,000`; });
document.querySelectorAll('[data-prefix]').forEach((button) => button.addEventListener('click', () => {
  $('prefix').value = button.dataset.prefix;
  $('prefix').dispatchEvent(new Event('input'));
  $('prefix').focus();
}));
$('generate-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (busy) return;
  $('notice').hidden = true;
  setBusy(true);
  const started = performance.now();
  try {
    const response = await api('generate', { prefix: $('prefix').value, topic: $('topic').value.trim() || null });
    lastText = response.text;
    $('empty-result').hidden = true;
    $('result').hidden = false;
    $('result').textContent = lastText || 'The model could not find a continuation. Try another prefix or add training texts.';
    $('copy-button').disabled = !lastText;
    $('result-meta').textContent = `${number.format(lastText.length)} characters · ${((performance.now() - started) / 1000).toFixed(1)} s`;
  } catch (error) { notify(error.message, true); }
  finally { setBusy(false); }
});
$('copy-button').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText(lastText); notify('Text copied.'); }
  catch { notify('Could not copy the text. Select it and copy it manually.', true); }
});
$('refresh-button').addEventListener('click', () => { if (!busy) refresh(true); });
$('replace').addEventListener('change', () => { $('replace-warning').hidden = !$('replace').checked; });
$('import-txt-button').addEventListener('click', () => {
  if (!busy) $('training-file').click();
});
$('training-file').addEventListener('change', () => {
  const file = $('training-file').files[0];
  if (!file || busy) return;
  if (!file.name.toLowerCase().endsWith('.txt')) {
    notify('Choose a file with the .txt extension.', true);
    $('training-file').value = '';
    return;
  }
  trainingFile = file;
  $('file-status').textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB. Will be uploaded as a file during training.`;
  $('file-status').hidden = false;
  $('remove-file-button').hidden = false;
});
$('remove-file-button').addEventListener('click', () => {
  trainingFile = null;
  $('training-file').value = '';
  $('file-status').hidden = true;
  $('remove-file-button').hidden = true;
});
$('train-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (busy) return;
  const texts = $('training-texts').value.split(/\r\n|[\r\n]/).map((text) => text.trim()).filter(Boolean);
  if (!texts.length && !trainingFile) {
    notify('Add at least one nonempty text.', true); return;
  }
  if ($('replace').checked && !window.confirm('Replace the current corpus with your texts? The saved model will also be replaced.')) return;
  setBusy(true, true);
  notify('Training is in progress. Progress is shown below.');
  try {
    if (trainingFile) {
      const form = new FormData();
      form.append('file', trainingFile);
      form.append('replace', String($('replace').checked));
      form.append('texts', JSON.stringify(texts));
      setStats(await streamTraining('train/file', form));
    } else {
      setStats(await streamTraining('train', { texts, replace: $('replace').checked }));
    }
    notify('Training complete. The model is ready for new stories.');
  } catch (error) { notify(error.message, true); }
  finally { setBusy(false); }
});
$('settings-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (busy) return;
  const settings = {
    tokenizer: $('tokenizer').value,
    n_gramm: Number($('n-gramm').value),
    min_frequency: Number($('min-frequency').value),
    max_length: Number($('max-length').value),
  };
  const rebuild = !currentSettings || settings.tokenizer !== currentSettings.tokenizer || settings.n_gramm !== currentSettings.n_gramm
    || settings.min_frequency !== currentSettings.min_frequency;
  setBusy(true, true);
  $('settings-button').firstChild.textContent = rebuild ? 'Rebuilding model… ' : 'Applying… ';
  notify(rebuild ? 'Retraining on the current corpus. Progress is shown on this page.' : 'Applying settings.');
  try {
    setStats(await streamTraining('settings', settings));
    notify('Settings applied. The model is ready to generate text.');
  } catch (error) { notify(error.message, true); }
  finally { setBusy(false); $('settings-button').firstChild.textContent = 'Apply settings '; }
});
refresh();

function updateTokenizerHint() {
  $('tokenizer-hint').textContent = $('tokenizer').value === 'character'
    ? 'Each token is one character. Useful for creating new words.'
    : 'Words, numbers, spaces, and punctuation are separate tokens. Generation uses words from the corpus.';
}
$('tokenizer').addEventListener('change', updateTokenizerHint);
