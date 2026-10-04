const $ = (id) => document.getElementById(id);
let busy = false;
let lastText = '';
let currentSettings = null;
let trainingFile = null;
const number = new Intl.NumberFormat('ru-RU');

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
    throw new Error('Нет связи с сервером. Убедитесь, что приложение запущено, и попробуйте снова.');
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    if (response.status === 503) throw new Error('Модель ещё загружается. Попробуйте немного позже.');
    const detail = body.detail;
    if (typeof detail === 'string') {
      if (detail.startsWith('No training texts found for topic:')) throw new Error('В корпусе нет текстов с этой темой. Попробуйте другую тему или оставьте поле пустым.');
      throw new Error(detail);
    }
    if (Array.isArray(detail)) throw new Error(detail.map((item) => item.msg).join('; '));
    throw new Error(`Не удалось выполнить запрос (${response.status}). Попробуйте снова.`);
  }
  return response.json();
}

function updateProgress(progress) {
  $('progress-stage').textContent = progress.stage;
  $('progress-percent').textContent = `${progress.percent}%`;
  $('progress-bar').value = progress.percent;
  $('progress-count').textContent = `${number.format(progress.completed)} / ${number.format(progress.total)}`;
  $('progress-time').textContent = `${progress.elapsed.toFixed(1)} сек. на этапе`;
}

async function streamTraining(path, data) {
  const panel = $('training-progress');
  panel.hidden = false;
  panel.classList.remove('failed');
  updateProgress({stage: data instanceof FormData ? 'Загрузка файла на сервер' : 'Ожидание обучения', percent: 0, completed: 0, total: 0, elapsed: 0});
  let result = null;
  try {
    const multipart = data instanceof FormData;
    const response = await fetch(`/api/v1/${path}/stream`, {
      method: 'POST', ...(multipart ? {} : {headers: {'Content-Type': 'application/json'}}),
      body: multipart ? data : JSON.stringify(data),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(typeof body.detail === 'string' ? body.detail : `Ошибка запроса (${response.status}). Проверьте значения настроек.`);
    }
    if (!response.body) throw new Error('Браузер не поддерживает потоковый ответ.');
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    try {
      while (true) {
        const {value, done} = await reader.read();
        buffer += decoder.decode(value, {stream: !done});
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
    } finally { await reader.cancel().catch(() => {}); reader.releaseLock(); }
    if (!result) throw new Error('Связь прервалась до завершения обучения. Обновите статистику, чтобы проверить состояние модели.');
    $('progress-stage').textContent = 'Готово';
    $('progress-percent').textContent = '100%';
    $('progress-bar').value = 100;
    return result;
  } catch (error) {
    panel.classList.add('failed');
    $('progress-stage').textContent = 'Обучение не подтверждено';
    throw error;
  }
}

function setStats(stats) {
  currentSettings = stats.settings;
  $('n-gramm').value = stats.settings.n_gramm;
  $('min-frequency').value = stats.settings.min_frequency;
  $('max-length').value = stats.settings.max_length;
  $('texts-count').textContent = number.format(stats.texts_count);
  $('vocab-size').textContent = number.format(stats.vocab_size);
  $('contexts-count').textContent = number.format(stats.contexts_count);
  $('status').className = 'status ready';
  $('status').replaceChildren(Object.assign(document.createElement('i')), document.createTextNode('Модель готова'));
}

async function refresh(showError = false) {
  $('refresh-button').disabled = true;
  try { setStats(await api('model')); }
  catch (error) {
    $('status').className = 'status offline';
    $('status').replaceChildren(document.createElement('i'), document.createTextNode('Сервер недоступен'));
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
  $('generate-button').firstChild.textContent = value && !training ? 'Создаём продолжение… ' : 'Продолжить текст ';
  $('train-button').firstChild.textContent = value && training ? 'Обучаем модель… ' : 'Обучить модель ';
}

$('prefix').addEventListener('input', () => { $('char-count').textContent = `${number.format($('prefix').value.length)} / 10 000`; });
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
    $('result').textContent = lastText || 'Модель не нашла продолжения. Попробуйте другое начало или добавьте тексты для обучения.';
    $('copy-button').disabled = !lastText;
    $('result-meta').textContent = `${number.format(lastText.length)} символов · ${((performance.now() - started) / 1000).toFixed(1)} сек.`;
  } catch (error) { notify(error.message, true); }
  finally { setBusy(false); }
});
$('copy-button').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText(lastText); notify('Текст скопирован.'); }
  catch { notify('Не удалось скопировать текст. Выделите его и скопируйте вручную.', true); }
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
    notify('Выберите файл с расширением .txt.', true);
    $('training-file').value = '';
    return;
  }
  trainingFile = file;
  $('file-status').textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} МБ. Будет отправлен файлом при обучении.`;
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
  const texts = $('training-texts').value.split(/\n\s*\n/).map((text) => text.trim()).filter(Boolean);
  if (!texts.length && !trainingFile) {
    notify('Добавьте хотя бы один непустой текст.', true); return;
  }
  if ($('replace').checked && !window.confirm('Заменить текущий корпус вашими текстами? Изменение действует до перезапуска сервера.')) return;
  setBusy(true, true);
  notify('Идёт обучение. Прогресс отображается ниже.');
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
    notify('Обучение завершено. Модель готова к новым историям.');
  } catch (error) { notify(error.message, true); }
  finally { setBusy(false); }
});
$('settings-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (busy) return;
  const settings = {
    n_gramm: Number($('n-gramm').value),
    min_frequency: Number($('min-frequency').value),
    max_length: Number($('max-length').value),
  };
  const rebuild = !currentSettings || settings.n_gramm !== currentSettings.n_gramm
    || settings.min_frequency !== currentSettings.min_frequency;
  setBusy(true, true);
  $('settings-button').firstChild.textContent = rebuild ? 'Пересчитываем модель… ' : 'Применяем… ';
  notify(rebuild ? 'Идёт переобучение на текущем корпусе. Прогресс отображается на странице.' : 'Применяем настройки.');
  try {
    setStats(await streamTraining('settings', settings));
    notify('Настройки применены. Модель готова к генерации.');
  } catch (error) { notify(error.message, true); }
  finally { setBusy(false); $('settings-button').firstChild.textContent = 'Применить настройки '; }
});
refresh();
