/* Tess Memory setup. One page, no build step, no third-party code.
   The server holds all state; this file draws it and sends what the person
   chooses. Text from the machine (paths, model names, log lines) is only ever
   set as text, never as markup. */
(() => {
  'use strict';

  const STEPS = [
    { id: 'sources', label: 'Sources' },
    { id: 'model', label: 'Model' },
    { id: 'embedding', label: 'Embeddings' },
    { id: 'agents', label: 'Connect agents' },
    { id: 'run', label: 'First run' },
  ];

  const MESSAGES = {
    token_required: 'This page was opened without its setup link. Use the full link printed in your terminal.',
    wrong_host: 'Open setup at the address printed in your terminal.',
    no_source_selected: 'Choose at least one source.',
    folder_not_found: 'That folder was not found. Give its full path.',
    invalid_key: 'That does not look like a key. Paste it again without spaces.',
    invalid_url: 'Enter a full address that starts with http:// or https://.',
    invalid_model: 'Enter a model name.',
    authentication_denied: 'The provider refused that key. Check it and try again.',
    connection_unavailable: 'The provider could not be reached. Check the address and your connection.',
    timeout: 'The provider did not answer in time. Try again.',
    rate_limited: 'The provider is limiting requests right now. Try again in a minute.',
    invalid_provider_response: 'The provider answered in a way that was not understood. Check the address and the model name.',
    model_or_endpoint_not_found: 'The provider does not know that model or address.',
    unexpected_redirect: 'The address redirects somewhere else. Use the final address.',
    needs_keychain: 'ChatGPT sign-in needs the system keychain, which is not available here. Use an API key instead.',
    needs_cryptography: 'ChatGPT sign-in needs the cryptography package. Use an API key instead.',
    loopback_unavailable: 'The sign-in could not listen on this machine. Close other sign-in windows and try again.',
    timed_out: 'The sign-in was not finished in time. Start it again.',
    access_denied: 'The sign-in was declined.',
    plan_use_not_granted: 'You signed in, but did not allow plan use, so the account cannot be used for requests. Sign in again and allow it, or use an API key.',
    no_models: 'The account lists no models that can be used.',
    sign_in_failed: 'The sign-in did not complete. Try again.',
    credential_unavailable: 'The key could not be kept in the keychain. Start setup with --no-keychain to keep keys in a private file instead.',
    download_failed: 'The download did not finish. Check your connection and try again; it resumes where it stopped.',
    insufficient_space: 'There is not enough free disk space for this model.',
    checksum_mismatch: 'The downloaded file did not match its published checksum, so it was discarded. Try again.',
    model_not_usable: 'The model downloaded but could not be loaded.',
    download_running: 'A download is already running.',
    incomplete: 'Some earlier steps are not finished yet.',
    settings_not_written: 'Save the settings first.',
    agent_command_not_found: 'The agent\'s command was not found on this machine. Use the manual configuration below.',
    agent_command_failed: 'The agent\'s command did not accept the registration.',
    guidance_not_written: 'The agent is connected, but its instruction file could not be written.',
    ingest_running: 'Ingestion is already running.',
    scheduler_already_installed: 'A scheduler is already installed on this machine and was left alone.',
    scheduler_install_failed: 'The background service could not be installed.',
    scheduler_remove_failed: 'The background service could not be turned off.',
    scheduler_unsupported: 'A background service cannot be installed on this system.',
  };

  const say = (error) => {
    const text = MESSAGES[error.code] || 'Something went wrong (' + error.code + ').';
    return error.detail && !MESSAGES[error.code] ? text + ' ' + error.detail : text;
  };

  let token = '';
  try {
    const given = new URLSearchParams(location.search).get('token');
    token = given || sessionStorage.getItem('tess-memory-setup-token') || '';
    if (given) {
      sessionStorage.setItem('tess-memory-setup-token', given);
      history.replaceState(null, '', '/');
    }
  } catch (_) { token = new URLSearchParams(location.search).get('token') || ''; }

  let S = null;                       // the server's state
  const ui = { step: 'sources', modelTab: 'chatgpt', embedTab: 'local', error: null, busy: null,
               sources: null, changeModel: false, signInUrl: null, written: null, finished: false, limit: 25 };

  async function api(path, body) {
    const response = await fetch(path, {
      method: body === undefined ? 'GET' : 'POST',
      headers: { 'X-Setup-Token': token, ...(body === undefined ? {} : { 'Content-Type': 'application/json' }) },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const payload = await response.json().catch(() => ({ error: 'invalid_provider_response' }));
    if (!response.ok) throw { code: payload.error || 'failed', detail: payload.detail || '' };
    return payload;
  }

  // Build an element: h('p', {class: 'hint'}, 'text', child, ...)
  function h(tag, attrs, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (value === false || value == null) continue;
      if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
      else if (key === 'class') node.className = value;
      else if (key === 'value') node.value = value;
      else if (key === 'checked' || key === 'disabled') node[key] = Boolean(value);
      else node.setAttribute(key, value === true ? '' : value);
    }
    for (const child of children.flat()) {
      if (child == null || child === false) continue;
      node.append(child.nodeType ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  const number = (value) => Number(value || 0).toLocaleString();
  const megabytes = (bytes) => (bytes / 1048576).toFixed(bytes < 104857600 ? 1 : 0) + ' MB';
  function ago(seconds) {
    if (!seconds) return '';
    const days = Math.floor((Date.now() / 1000 - seconds) / 86400);
    if (days <= 0) return 'active today';
    if (days === 1) return 'active yesterday';
    return days < 60 ? 'active ' + days + ' days ago' : 'active ' + Math.floor(days / 30) + ' months ago';
  }

  // Run an action that returns the new state; show its error beside the step.
  async function act(name, path, body, after) {
    if (ui.busy) return;
    ui.busy = name; ui.error = null; render();
    try {
      S = await api(path, body);
      ui.busy = null;
      if (after) after(S.result);
    } catch (error) {
      ui.busy = null; ui.error = error;
    }
    render();
  }

  const button = (label, name, onclick, kind = 'primary', extra = {}) =>
    h('button', { type: 'button', class: 'button button-' + kind + (ui.busy === name ? ' busy' : ''),
                  disabled: Boolean(ui.busy) || extra.disabled, onclick, ...extra.attrs }, label);

  const errorNote = () => ui.error ? h('p', { class: 'note bad', role: 'alert' }, say(ui.error)) : null;

  function reachable(id) {
    const model = S.model.connection;
    if (id === 'sources') return true;
    if (id === 'model') return S.sources_saved;
    if (id === 'embedding') return S.sources_saved && Boolean(model && model.verified);
    if (id === 'agents' || id === 'run') return Boolean(S.applied) && S.missing.length === 0;
    return false;
  }
  function complete(id) {
    const index = STEPS.findIndex((step) => step.id === id);
    const next = STEPS[index + 1];
    return next ? reachable(next.id) && (id !== 'agents' || ui.step === 'run' || ui.finished) : ui.finished;
  }
  function go(step) { ui.step = step; ui.error = null; render(); document.getElementById('main').focus(); window.scrollTo(0, 0); }

  // ---- step 1: sources ----------------------------------------------------

  function sourcesStep() {
    if (!ui.sources) ui.sources = S.sources.map((source) => ({ ...source }));
    const rows = ui.sources.map((source, index) => h('label', { class: 'row' },
      h('input', { type: 'checkbox', checked: source.enabled, onchange: (event) => { ui.sources[index].enabled = event.target.checked; render(); } }),
      h('span', { class: 'row-main' },
        h('span', { class: 'row-title' }, source.label), ' ',
        h('span', { class: 'row-sub mono' }, source.display_path)),
      h('span', { class: 'row-meta' }, h('strong', null, number(source.sessions)), source.sessions === 1 ? 'session file' : 'session files',
        source.newest ? h('span', null, ' · ' + ago(source.newest)) : null)));
    const kind = h('select', { id: 'folder-kind', 'aria-label': 'Kind of folder' },
      h('option', { value: 'claude-code' }, 'Claude Code projects folder'),
      h('option', { value: 'codex' }, 'Codex home folder'),
      h('option', { value: 'tess-core' }, 'Tess Core sessions folder'),
      h('option', { value: 'transcript' }, 'Folder of transcript files'));
    const path = h('input', { type: 'text', class: 'mono', id: 'folder-path', placeholder: '/full/path/to/folder', autocomplete: 'off', spellcheck: 'false' });
    const chosen = ui.sources.filter((source) => source.enabled);
    const total = chosen.reduce((sum, source) => sum + source.sessions, 0);
    return [
      h('p', { class: 'eyebrow' }, 'Step 1 of 5'),
      h('h1', null, 'What should go into ', h('mark', null, 'memory?')),
      h('p', { class: 'lede' }, 'Everything your agents have already done is sitting in these folders. Pick what should become memory. Nothing in them is changed. Tess Memory will then keep adding new sessions from these folders.'),
      h('div', { class: 'section' },
        rows.length ? h('div', { class: 'list' }, rows)
                    : h('p', { class: 'note' }, 'No Claude Code or Codex sessions were found in your home folder. Add a folder below.')),
      h('details', { class: 'section' },
        h('summary', { class: 'hint' }, 'Add another folder'),
        h('div', { class: 'panel' },
          h('div', { class: 'fields-2' },
            h('label', { class: 'field' }, h('span', null, 'Kind'), kind),
            h('label', { class: 'field' }, h('span', null, 'Folder'), path)),
          h('div', null, button('Add folder', 'folder', async () => {
            if (ui.busy) return;
            ui.busy = 'folder'; ui.error = null;
            try {
              const found = (await api('/api/sources/folder', { type: kind.value, path: path.value.trim() })).result;
              if (!ui.sources.some((source) => source.type === found.type && source.path === found.path)) ui.sources.push({ ...found, enabled: true });
            } catch (error) { ui.error = error; }
            ui.busy = null; render();
          }, 'outline')))),
      errorNote(),
      h('div', { class: 'actions' },
        button('Continue', 'sources', () => act('sources', '/api/sources',
          { sources: ui.sources.map(({ type, path: folder, enabled }) => ({ type, path: folder, enabled })) },
          () => { ui.sources = null; ui.step = 'model'; }), 'primary', { disabled: !chosen.length }),
        h('span', { class: 'hint' }, chosen.length ? number(total) + ' session files in ' + chosen.length + (chosen.length === 1 ? ' source' : ' sources') : 'Choose at least one source')),
    ];
  }

  // ---- step 2: model ------------------------------------------------------

  function modelStep() {
    const connection = S.model.connection;
    const picking = !connection || ui.changeModel;
    const head = [
      h('p', { class: 'eyebrow' }, 'Step 2 of 5'),
      h('h1', null, 'Connect a ', h('mark', null, 'provider.')),
      h('p', { class: 'lede' }, 'Tess Memory uses a model to distil your sessions into memories and to search them. Connect the account you want it to use.'),
    ];
    return picking ? [...head, ...connectModel(connection)] : [...head, ...chooseModels(connection)];
  }

  function connectModel(connection) {
    const chat = S.model.chatgpt;
    const tab = (id, title, sub) => h('button', { type: 'button', class: 'choice', 'aria-pressed': String(ui.modelTab === id),
      onclick: () => { ui.modelTab = id; ui.error = null; render(); } },
      h('span', { class: 'choice-title' }, title), h('span', { class: 'choice-sub' }, sub));
    let panel;
    if (ui.modelTab === 'chatgpt') {
      const attempt = chat.attempt || {};
      if (!chat.available) {
        panel = h('div', { class: 'panel' }, h('p', { class: 'note' }, MESSAGES[chat.reason] || 'ChatGPT sign-in is not available here.'));
      } else if (attempt.state === 'waiting') {
        panel = h('div', { class: 'panel' },
          h('p', { class: 'note' }, 'Waiting for you to finish signing in. The sign-in page opened in another tab.'),
          h('div', null,
            ui.signInUrl ? h('a', { class: 'button button-outline button-small', href: ui.signInUrl, target: '_blank', rel: 'noreferrer' }, 'Open the sign-in page again') : null, ' ',
            button('Cancel', 'chatgpt-cancel', () => act('chatgpt-cancel', '/api/model/chatgpt/cancel', {}), 'quiet')));
      } else {
        panel = h('div', { class: 'panel' },
          h('p', { class: 'hint' }, chat.disclosure),
          attempt.state === 'failed' ? h('p', { class: 'note bad', role: 'alert' }, say({ code: attempt.code })) : null,
          h('div', null, button('Continue with ChatGPT', 'chatgpt', () => {
            // Open the tab in the click itself so the browser allows it; point it at the sign-in once the link exists.
            const opened = window.open('', '_blank');
            act('chatgpt', '/api/model/chatgpt/start', {}, (result) => {
              ui.signInUrl = result.url;
              if (!result.url) { ui.changeModel = false; if (opened) opened.close(); return; }  // already signed in
              if (opened) opened.location = result.url;
            }).then(() => { if (ui.error && opened) opened.close(); });
          })));
      }
    } else if (ui.modelTab === 'key') {
      const provider = h('select', { id: 'key-provider' }, h('option', { value: 'anthropic' }, 'Anthropic'), h('option', { value: 'openai' }, 'OpenAI'));
      const key = h('input', { type: 'password', class: 'mono', id: 'key-value', autocomplete: 'off', spellcheck: 'false', placeholder: 'Paste your API key' });
      panel = h('form', { class: 'panel', onsubmit: (event) => { event.preventDefault(); act('key', '/api/model/key', { kind: provider.value, api_key: key.value }, () => { ui.changeModel = false; }); } },
        h('div', { class: 'fields-2' },
          h('label', { class: 'field' }, h('span', null, 'Provider'), provider),
          h('label', { class: 'field' }, h('span', null, 'API key'), key)),
        h('p', { class: 'hint' }, keyHint() + ' You choose the models on the next screen, from the ones this account offers.'),
        h('div', null, h('button', { type: 'submit', class: 'button button-primary' + (ui.busy === 'key' ? ' busy' : ''), disabled: Boolean(ui.busy) }, 'Connect')));
    } else {
      const url = h('input', { type: 'url', class: 'mono', id: 'endpoint-url', autocomplete: 'off', spellcheck: 'false', placeholder: 'http://localhost:11434/v1' });
      const key = h('input', { type: 'password', class: 'mono', id: 'endpoint-key', autocomplete: 'off', spellcheck: 'false', placeholder: 'Leave empty if none is needed' });
      panel = h('form', { class: 'panel', onsubmit: (event) => { event.preventDefault(); act('key', '/api/model/key', { kind: 'compatible', base_url: url.value, api_key: key.value }, () => { ui.changeModel = false; }); } },
        h('label', { class: 'field' }, h('span', null, 'Address'), url, h('small', null, 'The base address of any service that speaks the OpenAI chat API, including a model on your own machine.')),
        h('label', { class: 'field' }, h('span', null, 'API key'), key),
        h('p', { class: 'hint' }, keyHint() + ' You choose the models on the next screen, from the ones this account offers.'),
        h('div', null, h('button', { type: 'submit', class: 'button button-primary' + (ui.busy === 'key' ? ' busy' : ''), disabled: Boolean(ui.busy) }, 'Connect')));
    }
    return [
      h('div', { class: 'section choices' },
        tab('chatgpt', 'ChatGPT account', 'Sign in and use your ChatGPT plan. No key to manage.'),
        tab('key', 'API key', 'An Anthropic or OpenAI key, billed to your API account.'),
        tab('endpoint', 'Your own endpoint', 'Any OpenAI-compatible service, or a local model.')),
      panel,
      errorNote(),
      connection ? h('div', { class: 'actions' }, button('Keep the current connection', 'keep', () => { ui.changeModel = false; ui.error = null; render(); }, 'quiet')) : null,
    ];
  }

  const keyHint = () => S.secrets === 'keychain'
    ? 'The key goes straight into your system keychain. It is not written to any settings file.'
    : 'The key is kept in a file only you can read, beside the settings. No system keychain is available here.';

  function chooseModels(connection) {
    const models = connection.models || [];
    // A real menu of the listed models, with a way to type one that is not listed.
    const OTHER = '\u0000other';
    const field = (id, value) => {
      const text = h('input', { type: 'text', class: 'mono', id: id + '-text', value: value || '', autocomplete: 'off', spellcheck: 'false',
        placeholder: 'Type the model name', 'aria-label': 'Model name' });
      if (!models.length) return { node: text, read: () => text.value.trim() };
      const listed = models.includes(value);
      const menu = h('select', { id, class: 'mono' },
        models.map((model) => h('option', { value: model }, model)),
        h('option', { value: OTHER }, 'Another model…'));
      menu.value = listed || !value ? (value || models[0]) : OTHER;
      text.hidden = menu.value !== OTHER;
      menu.addEventListener('change', () => { text.hidden = menu.value !== OTHER; if (!text.hidden) { text.value = ''; text.focus(); } });
      return { node: [menu, text], read: () => (menu.value === OTHER ? text.value.trim() : menu.value) };
    };
    const distill = field('distill-model', connection.distill_model || models[0]);
    const search = field('search-model', connection.search_model || connection.distill_model || models[0]);
    const checks = connection.checks || {};
    const failed = Object.entries(checks).filter(([, [ok]]) => !ok);
    const who = connection.email || connection.base_url || '';
    return [
      h('div', { class: 'section' },
        h('p', { class: 'note ok' }, 'Connected: ' + connection.label + (who ? ' (' + who + ')' : '') + '. ',
          models.length ? number(models.length) + ' models available.' : 'This service does not list its models, so type their names.')),
      h('div', { class: 'panel' },
        h('label', { class: 'field' }, h('span', null, 'Model for distilling sessions'), distill.node,
          h('small', null, 'Reads every session once and decides what is worth remembering. This is where nearly all the usage goes, so it sets both the quality of the memory and what it costs.')),
        h('label', { class: 'field' }, h('span', null, 'Model for searching'), search.node,
          h('small', null, 'Runs when an agent searches: a few short calls that judge the results and look further. An agent waits on it, so speed matters here.'))),
      failed.length ? h('p', { class: 'note bad', role: 'alert' }, 'A test request failed: ' + failed.map(([role, [, detail]]) => role + ': ' + detail).join(' · ')) : null,
      errorNote(),
      h('div', { class: 'actions' },
        button(connection.verified ? 'Continue' : 'Test and continue', 'choose', () => {
          const body = { distill_model: distill.read(), search_model: search.read() };
          if (connection.verified && body.distill_model === connection.distill_model && body.search_model === connection.search_model) return go('embedding');
          act('choose', '/api/model/choose', body, () => { if (S.model.connection.verified) ui.step = 'embedding'; });
        }),
        button('Use a different connection', 'change', () => { ui.changeModel = true; ui.error = null; render(); }, 'quiet'),
        ui.busy === 'choose' ? h('span', { class: 'hint' }, 'Sending one short request to each model…') : null),
    ];
  }

  // ---- step 3: embeddings -------------------------------------------------

  function embeddingStep() {
    const embedding = S.embedding;
    const choice = embedding.choice;
    const tab = (id, title, sub) => h('button', { type: 'button', class: 'choice', 'aria-pressed': String(ui.embedTab === id),
      onclick: () => { ui.embedTab = id; ui.error = null; render(); } },
      h('span', { class: 'choice-title' }, title), h('span', { class: 'choice-sub' }, sub));
    let panel;
    if (ui.embedTab === 'local') {
      const job = embedding.job || {};
      const active = job.state === 'downloading' || job.state === 'verifying';
      panel = h('div', { class: 'section list' }, embedding.local.map((model) => {
        const current = choice && choice.mode === 'local' && choice.name === model.name;
        const mine = job.name === model.name;
        return h('div', { class: 'row' },
          h('div', { class: 'row-main' },
            h('div', { class: 'row-title' }, model.name, ' ',
              model.recommended ? h('span', { class: 'badge pick' }, 'Recommended') : null, ' ',
              current ? h('span', { class: 'badge ok' }, 'In use') : null),
            h('div', { class: 'row-sub' }, megabytes(model.bytes) + ' download · ' + model.dims + ' dimensions · ' + model.license.toUpperCase() + ' licence'),
            mine && active ? h('div', { class: 'section' },
              h('progress', { id: 'download-bar', max: job.total || 1, value: job.done || 0 }),
              h('div', { class: 'progress-line' },
                h('span', { id: 'download-text' }, job.state === 'verifying' ? 'Checking the model…' : megabytes(job.done || 0) + ' of ' + megabytes(job.total || 0)),
                button('Cancel', 'download-cancel', () => act('download-cancel', '/api/embedding/local/cancel', {}), 'quiet'))) : null,
            mine && job.state === 'failed' ? h('p', { class: 'note bad', role: 'alert' }, say({ code: job.code })) : null),
          current || (mine && active) ? null
            : model.installed ? button('Use this model', 'use-' + model.name, () => act('use-' + model.name, '/api/embedding/local/use', { name: model.name }), 'outline', { attrs: { class: 'button button-outline button-small' } })
              : button('Download', 'dl-' + model.name, () => act('dl-' + model.name, '/api/embedding/local/download', { name: model.name }), 'outline', { disabled: active, attrs: { class: 'button button-outline button-small' } }));
      }));
    } else {
      const google = ui.embedTab === 'google';
      const preset = embedding.presets[google ? 'google' : 'compatible'];
      const url = h('input', { type: 'url', class: 'mono', autocomplete: 'off', spellcheck: 'false', placeholder: 'https://api.example.com/v1' });
      const model = h('input', { type: 'text', class: 'mono', value: preset.model, autocomplete: 'off', spellcheck: 'false', placeholder: 'Embedding model name' });
      const key = h('input', { type: 'password', class: 'mono', autocomplete: 'off', spellcheck: 'false', placeholder: google ? 'Paste your Gemini API key' : 'Leave empty if none is needed' });
      panel = h('form', { class: 'panel', onsubmit: (event) => { event.preventDefault(); act('hosted', '/api/embedding/hosted',
          { preset: google ? 'google' : 'compatible', base_url: google ? undefined : url.value, model: model.value, api_key: key.value }); } },
        google ? null : h('label', { class: 'field' }, h('span', null, 'Address'), url, h('small', null, 'The base address of a service with an OpenAI-style /embeddings endpoint.')),
        h('div', { class: 'fields-2' },
          h('label', { class: 'field' }, h('span', null, 'Model'), model),
          h('label', { class: 'field' }, h('span', null, 'API key'), key)),
        h('p', { class: 'hint' }, 'The text of your sessions is sent to this service to be embedded. ' + keyHint()),
        h('div', null, h('button', { type: 'submit', class: 'button button-primary' + (ui.busy === 'hosted' ? ' busy' : ''), disabled: Boolean(ui.busy) }, 'Connect and test')));
    }
    const summary = !choice ? null : choice.mode === 'local'
      ? 'Using ' + choice.name + ' on this machine.'
      : 'Using ' + choice.model + ' from ' + choice.label + ' (' + choice.dims + ' dimensions).';
    return [
      h('p', { class: 'eyebrow' }, 'Step 3 of 5'),
      h('h1', null, 'Connect an ', h('mark', null, 'embedding model.')),
      h('p', { class: 'lede' }, 'An embedding model turns text into numbers, so a search can match on meaning and not just words. You can change it later, but every memory then has to be embedded again.'),
      h('div', { class: 'section choices' },
        tab('local', 'On this machine', 'Private and free. A small model is downloaded once.'),
        tab('google', 'Google', 'Gemini embeddings, with your own API key.'),
        tab('endpoint', 'Your own endpoint', 'Any OpenAI-compatible embeddings service.')),
      panel,
      summary ? h('p', { class: 'section note ok' }, summary) : null,
      S.existing_settings ? h('p', { class: 'section note' }, 'A settings file already exists at ', h('span', { class: 'mono' }, S.settings), '. Saving replaces it and keeps the current one beside it as a dated copy.') : null,
      errorNote(),
      h('div', { class: 'actions' },
        button('Save settings and continue', 'apply', () => act('apply', '/api/apply', {}, (result) => { ui.written = result; ui.step = 'agents'; }),
          'primary', { disabled: !choice })),
    ];
  }

  // ---- step 4: agents -----------------------------------------------------

  function agentsStep() {
    const snippet = JSON.stringify(S.agents.entry, null, 2);
    const copy = (textToCopy) => async (event) => {
      try { await navigator.clipboard.writeText(textToCopy); event.target.textContent = 'Copied'; } catch (_) { event.target.textContent = 'Select the text to copy it'; }
    };
    const rows = S.agents.targets.map((target) => {
      const ready = target.current && target.guidance;
      const label = !target.registered ? 'Connect' : !target.current ? 'Point it here' : 'Add instructions';
      return h('div', { class: 'row' },
        h('div', { class: 'row-main' },
          h('div', { class: 'row-title' }, target.label, ' ',
            ready ? h('span', { class: 'badge ok' }, 'Connected')
              : target.current ? h('span', { class: 'badge' }, 'Connected, no instructions yet')
                : target.registered ? h('span', { class: 'badge' }, 'Connected to another store' + (target.registered_as && target.registered_as !== 'tess-memory' ? ' as ' + target.registered_as : '')) : null),
          h('div', { class: 'row-sub mono' }, target.display_path),
          h('div', { class: 'row-sub' }, target.guidance
            ? 'Usage instructions are in ' + target.guidance_file + '.'
            : 'Connecting also adds usage instructions to ' + target.guidance_file + '.'),
          target.cli || target.current ? null : h('div', { class: 'row-sub' }, 'Its command was not found on this machine; use the configuration below.')),
        ready || !(target.cli || target.current) ? null
          : button(label, 'reg-' + target.id, () => act('reg-' + target.id, '/api/agents/register', { id: target.id }), 'outline', { attrs: { class: 'button button-outline button-small' } }));
    });
    return [
      h('p', { class: 'eyebrow' }, 'Step 4 of 5'),
      h('h1', null, 'Give your agents ', h('mark', null, 'the memory.')),
      h('p', { class: 'lede' }, 'Connect any agent on this machine, whether or not its own sessions are being ingested. Each agent you connect gets twelve tools: search what was done before, pick up a project where it was left, and keep a plan that outlasts the session. Connecting also adds a short set of instructions to the agent\'s own instruction file, so it reaches for them without being asked.'),
      h('p', { class: 'section note ok' }, 'Settings saved to ', h('span', { class: 'mono' }, S.settings), '.',
        ui.written && ui.written.backup ? [' The earlier file was kept as ', h('span', { class: 'mono' }, ui.written.backup), '.'] : null),
      rows.length ? h('div', { class: 'section' }, h('h2', null, 'Agents on this machine'), h('div', { class: 'list' }, rows)) : null,
      h('div', { class: 'section' },
        h('h2', null, 'Any other MCP client'),
        h('p', { class: 'hint' }, 'Add this server to the client\'s configuration, and the instructions below to the file it reads its instructions from.'),
        h('div', { class: 'section' }, h('pre', { class: 'block' }, snippet)),
        h('div', { class: 'section' }, h('button', { type: 'button', class: 'button button-outline button-small', onclick: copy(snippet) }, 'Copy configuration')),
        h('details', { class: 'section' },
          h('summary', { class: 'hint' }, 'The instructions agents are given'),
          h('div', { class: 'section' }, h('pre', { class: 'block log guidance' }, S.agents.guidance)),
          h('div', { class: 'section' }, h('button', { type: 'button', class: 'button button-outline button-small', onclick: copy(S.agents.guidance) }, 'Copy instructions')))),
      errorNote(),
      h('div', { class: 'actions' }, button('Continue', 'to-run', () => go('run'))),
    ];
  }

  // ---- step 5: first run --------------------------------------------------

  function runStep() {
    const ingest = S.ingest || {};
    const scheduler = S.scheduler;
    const total = S.sources.filter((source) => source.enabled).reduce((sum, source) => sum + source.sessions, 0);
    const foreign = scheduler.installed && !scheduler.ours;       // someone else's scheduler: never touched
    const canKeep = scheduler.supported && !foreign;
    if (ui.keep === undefined) ui.keep = scheduler.wanted;
    const keeping = canKeep && (scheduler.ours || ui.keep);
    const idle = !ingest.state || ingest.state === 'idle';
    const amount = (value, label, sub) => h('label', { class: 'row' },
      h('input', { type: 'radio', name: 'limit', checked: ui.limit === value, onchange: () => { ui.limit = value; } }),
      h('span', { class: 'row-main' }, h('span', { class: 'row-title' }, label), h('span', { class: 'row-sub' }, ' ' + sub)));
    if (ui.limit !== null && total <= ui.limit) ui.limit = null;   // nothing smaller than everything to offer
    const amounts = h('div', { class: 'list' },
      total > 25 ? amount(25, '25 sessions', 'A quick first pass to see it work.') : null,
      total > 200 ? amount(200, '200 sessions', 'A solid start. Takes a while and costs model calls.') : null,
      amount(null, 'Everything', 'Up to ' + number(total) + ' session files.' + (total > 200 ? ' Can run for hours on a long history.' : '')));
    const begin = () => act('ingest', '/api/ingest/start', { limit: keeping ? 25 : ui.limit, keep_current: canKeep ? Boolean(ui.keep) : false });

    let body;
    if (ingest.state === 'running') {
      body = h('div', { class: 'section panel' },
        h('progress', { id: 'ingest-bar', max: ingest.total || 1, value: ingest.total ? ingest.done : undefined }),
        h('div', { class: 'progress-line' },
          h('span', { id: 'ingest-text' }, number(ingest.done) + (ingest.total ? ' of ' + number(ingest.total) : '') + ' sessions ingested'),
          button('Stop ingesting', 'ingest-stop', () => act('ingest-stop', '/api/ingest/stop', {}), 'outline', { attrs: { class: 'button button-outline button-small' } })),
        h('pre', { class: 'block log', id: 'ingest-log' }, (ingest.lines || []).join('\n')),
        h('p', { class: 'hint' }, keeping ? 'When this first batch is in, the background service takes over and ingests the rest.' : 'You can leave this running. What is finished is kept if it stops.'));
    } else {
      const stats = ingest.stats;
      body = [
        ingest.state === 'finished' && stats ? h('p', { class: 'section note ok' },
          number(stats.distilled) + ' sessions distilled' + (stats.extractive ? ', ' + number(stats.extractive) + ' indexed without a summary' : '')
          + (stats.unchanged ? ', ' + number(stats.unchanged) + ' already up to date' : '')
          + (stats.quiet_deferred || stats.deferred_active ? ', ' + number((stats.quiet_deferred || 0) + (stats.deferred_active || 0)) + ' still in use and left for later' : '')
          + (stats.errors ? ', ' + number(stats.errors) + ' with errors' : '') + '.') : null,
        ingest.state === 'failed' ? h('div', { class: 'section' },
          h('p', { class: 'note bad', role: 'alert' }, 'Ingestion stopped with an error. The last lines it printed:'),
          h('div', { class: 'section' }, h('pre', { class: 'block log' }, (ingest.lines || []).join('\n')))) : null,
        ingest.state === 'stopped' ? h('p', { class: 'section note' }, 'Stopped. What was finished is kept.') : null,
        idle ? h('div', { class: 'section' },
          canKeep ? h('label', { class: 'row list' },
            h('input', { type: 'checkbox', checked: ui.keep, onchange: (event) => { ui.keep = event.target.checked; render(); } }),
            h('span', { class: 'row-main' },
              h('span', { class: 'row-title' }, 'Keep ingesting in the background'),
              h('span', { class: 'row-sub' }, ' Recommended. A small service that starts when you log in.'))) : null,
          keeping ? h('p', { class: 'section hint' }, 'Setup ingests 25 sessions now, so you can see it work. The background service then ingests the rest of your history, up to ' + number(total) + ' session files, and keeps going as you work. A long history takes hours and uses your model plan; that is the cost of giving your agents a memory, and it is paid once.')
            : h('div', { class: 'section' }, h('h2', null, 'How much to ingest now'), amounts),
          h('div', { class: 'actions' }, button('Start ingesting', 'ingest', begin)))
          : keeping ? null : h('div', { class: 'section' }, h('h2', null, 'Ingest more'), amounts,
              h('div', { class: 'actions' }, button('Ingest more', 'ingest', begin))),
      ];
    }

    const rules = h('ul', { class: 'rules hint' },
      h('li', null, 'Every 5 minutes it looks for new or changed sessions.'),
      h('li', null, 'A session is ingested once it has been still for 15 minutes, so a conversation is not distilled mid-thought.'),
      h('li', null, 'Project profiles refresh hourly, weekly digests daily.'),
      h('li', null, 'At 10 PM (' + scheduler.timezone + ') one pass retries whatever failed that day.'));
    let keep;
    if (!scheduler.supported) {
      keep = [h('p', { class: 'hint' }, 'To ingest new sessions as they finish, keep this command running under your service manager:'),
              h('div', { class: 'section' }, h('pre', { class: 'block' }, scheduler.command))];
    } else if (foreign) {
      keep = h('p', { class: 'note' }, 'A Tess Memory scheduler is already installed on this machine and was left alone. Add this store to its configuration to keep it current.');
    } else if (scheduler.ours) {
      const line = scheduler.running
        ? 'On.' + (scheduler.state === 'running' ? ' Ingesting now.' : '') + (scheduler.pending ? ' ' + number(scheduler.pending) + ' sessions still to ingest.' : '')
        : 'Installed and starting.';
      keep = [h('p', { class: 'note ok' }, line), rules,
              h('div', { class: 'actions' }, button('Turn it off', 'scheduler-off', () => act('scheduler-off', '/api/scheduler/remove', {}, () => { ui.keep = false; }), 'quiet'))];
    } else {
      keep = [scheduler.error ? h('p', { class: 'note bad', role: 'alert' }, say(scheduler.error)) : null,
              h('p', { class: 'hint' }, idle ? (ui.keep ? 'It is turned on when the first batch has been ingested.' : 'Off. New sessions are only ingested when you run ingestion yourself.')
                                              : 'Off. New sessions are only ingested when you run ingestion yourself.'),
              idle && !ui.keep ? null : rules,
              idle ? null : h('div', { class: 'actions' }, button('Turn it on', 'scheduler', () => act('scheduler', '/api/scheduler/install', {}, () => { ui.keep = true; }), 'outline'))];
    }
    return [
      h('p', { class: 'eyebrow' }, 'Step 5 of 5'),
      h('h1', null, 'Ingest what you have ', h('mark', null, 'already done.')),
      h('p', { class: 'lede' }, 'Tess Memory ingests your past sessions, distilling each one into memories, then keeps up as you work. Sessions that changed in the last fifteen minutes are left until they go quiet.'),
      body,
      h('div', { class: 'section' }, h('h2', null, 'Keeping it current'), keep),
      errorNote(),
      h('div', { class: 'actions' }, button('Finish setup', 'quit', () => act('quit', '/api/quit', {}, () => { ui.finished = true; }), 'primary', { disabled: idle })),
    ];
  }

  function finished() {
    const running = S.ingest && S.ingest.state === 'running';
    return [
      h('div', { class: 'done-mark', 'aria-hidden': 'true' }, '✓'),
      h('h1', null, 'Your agents ', h('mark', null, 'remember now.')),
      h('p', { class: 'lede' }, 'Setup is finished and this page has stopped. You can close the tab.'),
      running ? h('p', { class: 'section note' }, 'The first ingestion is still running in the background and will finish on its own.') : null,
      S.scheduler.ours ? h('p', { class: 'section note ok' }, 'The background service is on. It ingests the rest of your history and every new session from here.') : null,
      h('div', { class: 'section' },
        h('h2', null, 'What to do next'),
        h('p', { class: 'hint' }, 'Start a new session in a connected agent and ask it to pick up where you left off. To change anything later, run ', h('code', null, 'tess-memory setup'), ' again, or edit ', h('span', { class: 'mono' }, S.settings), '.')),
    ];
  }

  // ---- drawing ------------------------------------------------------------

  function render() {
    const main = document.getElementById('main');
    if (!S) return;
    // Where this store lives: ~/.tess-memory normally. A folder outside the
    // home directory is shown by its last two names, with the full path on hover.
    const parts = S.display_home.split('/').filter(Boolean);
    const short = S.display_home.startsWith('~') || parts.length <= 3 ? S.display_home : '…/' + parts.slice(-2).join('/');
    const where = document.getElementById('top-home');
    where.replaceChildren(h('span', { class: 'top-home-label' }, 'Store'), ' ' + short);
    where.title = S.home;
    if (!ui.finished && !reachable(ui.step)) ui.step = STEPS.filter((step) => reachable(step.id)).pop().id;
    const steps = document.getElementById('steps');
    steps.replaceChildren(...STEPS.map((step, index) => h('li', null, h('button', { type: 'button',
      class: 'step' + (complete(step.id) ? ' done' : ''), 'aria-current': !ui.finished && ui.step === step.id ? 'step' : null,
      disabled: ui.finished || !reachable(step.id), onclick: () => go(step.id) },
      h('span', { class: 'step-n' }, complete(step.id) ? '✓' : index + 1), h('span', { class: 'step-label' }, step.label)))));
    const shown = ui.finished ? 'finished' : ui.step;
    if (shown !== render.shown) { render.shown = shown; window.scrollTo(0, 0); }
    const draw = ui.finished ? finished : { sources: sourcesStep, model: modelStep, embedding: embeddingStep, agents: agentsStep, run: runStep }[ui.step];
    main.replaceChildren(...draw().flat().filter(Boolean));
  }

  // While something is in flight, ask for the state again. A change of phase
  // redraws the step; otherwise only the progress figures are touched, so
  // nothing the person is typing is lost.
  const phase = (state) => JSON.stringify([state.model.chatgpt.attempt.state, (state.embedding.job || {}).state, (state.ingest || {}).state,
                                           Boolean(state.model.connection), Boolean(state.embedding.choice),
                                           state.scheduler.ours, state.scheduler.running, state.scheduler.state, Boolean(state.scheduler.error)]);
  const inFlight = (state) => state.model.chatgpt.attempt.state === 'waiting'
    || ['downloading', 'verifying'].includes((state.embedding.job || {}).state) || (state.ingest || {}).state === 'running'
    || (ui.step === 'run' && state.scheduler.supported && (state.scheduler.ours || (state.scheduler.wanted && (state.ingest || {}).state === 'finished' && !state.scheduler.installed && !state.scheduler.error)));

  async function poll() {
    if (!S || ui.busy || ui.finished || !inFlight(S)) return;
    let next;
    try { next = await api('/api/state'); } catch (_) { return; }
    if (ui.busy) return;
    const changed = phase(next) !== phase(S);
    S = next;
    if (changed) {
      if (S.model.connection && ui.step === 'model') ui.changeModel = false;
      return render();
    }
    const job = S.embedding.job || {};
    const bar = document.getElementById('download-bar');
    if (bar) {
      bar.max = job.total || 1; bar.value = job.done || 0;
      document.getElementById('download-text').textContent = megabytes(job.done || 0) + ' of ' + megabytes(job.total || 0);
    }
    const ingest = S.ingest || {};
    const ingestBar = document.getElementById('ingest-bar');
    if (ingestBar) {
      if (ingest.total) { ingestBar.max = ingest.total; ingestBar.value = ingest.done; }
      document.getElementById('ingest-text').textContent = number(ingest.done) + (ingest.total ? ' of ' + number(ingest.total) : '') + ' sessions ingested';
      const log = document.getElementById('ingest-log');
      log.textContent = (ingest.lines || []).join('\n');
      log.scrollTop = log.scrollHeight;
    }
  }

  async function start() {
    try {
      S = await api('/api/state');
    } catch (error) {
      document.getElementById('main').replaceChildren(
        h('h1', null, 'Open setup from ', h('mark', null, 'its link.')),
        h('p', { class: 'lede' }, say(error)));
      return;
    }
    const firstOpen = STEPS.filter((step) => reachable(step.id)).pop();
    ui.step = firstOpen ? firstOpen.id : 'sources';
    if (ui.step === 'agents' && S.ingest.state !== 'idle') ui.step = 'run';
    render();
    setInterval(poll, 1500);
  }

  start();
})();
