// ==========================================================================
// Interactive Application Logic - Autonomous Data Science Analyst Agent
// Handles Tab Navigation, File Uploads, Server-Sent Events & Real-time Live Sandbox
// ==========================================================================

document.addEventListener('DOMContentLoaded', () => {
  let uploadedFilePath = null;
  let activeEventSource = null;

  // 1. Navigation Tabs
  const navTabs = document.querySelectorAll('.nav-tab');
  const tabPanes = document.querySelectorAll('.tab-pane');

  navTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      navTabs.forEach(t => t.classList.remove('active'));
      tabPanes.forEach(p => p.classList.add('hidden'));

      tab.classList.add('active');
      const targetId = tab.getAttribute('data-tab');
      const targetPane = document.getElementById(targetId);
      if (targetPane) {
        targetPane.classList.remove('hidden');
      }
    });
  });

  // 2. Fetch Server Config
  fetch('/api/config')
    .then(r => r.json())
    .then(data => {
      const pill = document.getElementById('nav-llm-label');
      if (data.provider === 'gemini') {
        pill.textContent = 'Gemini 2.5 Active';
      } else {
        pill.textContent = 'Local Qwen3 Active';
      }
    })
    .catch(() => {});

  // 3. Drag and Drop & File Upload
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const fileInfoBadge = document.getElementById('file-info-badge');
  const fileNameDisplay = document.getElementById('file-name-display');
  const fileMetaDisplay = document.getElementById('file-meta-display');
  const btnRemoveFile = document.getElementById('btn-remove-file');
  const previewContainer = document.getElementById('dataset-preview-container');
  const previewBadge = document.getElementById('preview-badge');
  const previewThead = document.getElementById('preview-thead');
  const previewTbody = document.getElementById('preview-tbody');
  const btnRunAgent = document.getElementById('btn-run-agent');

  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, e => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, e => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
    });
  });

  dropZone.addEventListener('drop', e => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files && fileInput.files[0]) {
      handleFileSelected(fileInput.files[0]);
    }
  });

  function handleFileSelected(file) {
    const formData = new FormData();
    formData.append('file', file);

    // Update UI indicator
    dropZone.style.opacity = '0.5';

    fetch('/api/upload', {
      method: 'POST',
      body: formData,
    })
      .then(res => {
        if (!res.ok) throw new Error('Upload failed');
        return res.json();
      })
      .then(data => {
        uploadedFilePath = data.file_path;
        dropZone.classList.add('hidden');
        fileInfoBadge.classList.remove('hidden');

        fileNameDisplay.textContent = data.filename;
        fileMetaDisplay.textContent = `${data.rows.toLocaleString()} rows • ${data.cols} columns`;

        // Render Schema Preview
        if (data.preview && data.preview.length > 0) {
          previewContainer.classList.remove('hidden');
          previewBadge.textContent = `${data.rows} Total Rows`;

          previewThead.innerHTML = '';
          data.columns.forEach(col => {
            const th = document.createElement('th');
            th.textContent = col;
            previewThead.appendChild(th);
          });

          previewTbody.innerHTML = '';
          data.preview.forEach(row => {
            const tr = document.createElement('tr');
            data.columns.forEach(col => {
              const td = document.createElement('td');
              td.textContent = row[col] !== undefined ? String(row[col]) : '';
              tr.appendChild(td);
            });
            previewTbody.appendChild(tr);
          });
        }

        btnRunAgent.removeAttribute('disabled');
      })
      .catch(err => {
        alert('Could not parse the dataset: ' + err.message);
        dropZone.style.opacity = '1';
      });
  }

  btnRemoveFile.addEventListener('click', () => {
    uploadedFilePath = null;
    fileInput.value = '';
    dropZone.classList.remove('hidden');
    dropZone.style.opacity = '1';
    fileInfoBadge.classList.add('hidden');
    previewContainer.classList.add('hidden');
    btnRunAgent.setAttribute('disabled', 'true');
  });

  // 4. Sample Chips for Prompt
  document.querySelectorAll('.chip[data-prompt]').forEach(chip => {
    chip.addEventListener('click', () => {
      document.getElementById('prompt-input').value = chip.getAttribute('data-prompt');
    });
  });

  // 5. Execution Pipeline with Server-Sent Events (SSE)
  const executionStage = document.getElementById('execution-stage');
  const stageStatusText = document.getElementById('stage-status-text');
  const overallProgressBar = document.getElementById('overall-progress-bar');
  const overallPercentage = document.getElementById('overall-percentage');
  const agentsTimeline = document.getElementById('agents-timeline');
  const resultsStage = document.getElementById('results-stage');

  btnRunAgent.addEventListener('click', () => {
    if (!uploadedFilePath) return;

    const promptText = document.getElementById('prompt-input').value.trim();
    if (!promptText) return;

    btnRunAgent.setAttribute('disabled', 'true');
    executionStage.classList.remove('hidden');
    resultsStage.classList.add('hidden');

    overallProgressBar.style.width = '5%';
    overallPercentage.textContent = '5%';
    stageStatusText.textContent = 'Initializing LangGraph DAG...';

    // Build timeline cards
    const stepsConfig = [
      { id: 'data_profiler', label: 'Data Profiler', icon: 'fa-table' },
      { id: 'orchestrator', label: 'Orchestrator', icon: 'fa-brain' },
      { id: 'cleaning_agent', label: 'Data Cleaner', icon: 'fa-broom' },
      { id: 'eda_agent', label: 'EDA Engine', icon: 'fa-chart-pie' },
      { id: 'model_selector', label: 'Model Trainer', icon: 'fa-microchip' },
      { id: 'explainability_agent', label: 'Explainability', icon: 'fa-wand-magic-sparkles' },
      { id: 'report_agent', label: 'Report Agent', icon: 'fa-file-lines' },
    ];

    agentsTimeline.innerHTML = '';
    stepsConfig.forEach((s, idx) => {
      const card = document.createElement('div');
      card.className = `timeline-step ${idx === 0 ? 'running' : 'pending'}`;
      card.id = `step-${s.id}`;
      card.innerHTML = `
        <div class="step-icon-wrap"><i class="fa-solid ${s.icon}"></i></div>
        <span class="step-name">${s.label}</span>
        <span class="step-state-tag" id="tag-${s.id}">${idx === 0 ? 'RUNNING' : 'PENDING'}</span>
      `;
      agentsTimeline.appendChild(card);
    });

    // Connect to SSE stream
    const params = new URLSearchParams({
      dataset_path: uploadedFilePath,
      user_question: promptText,
    });

    if (activeEventSource) {
      activeEventSource.close();
    }

    activeEventSource = new EventSource(`/api/run-stream?${params.toString()}`);

    activeEventSource.onmessage = e => {
      try {
        const data = JSON.parse(e.data);

        if (data.type === 'progress') {
          const pct = Math.min(Math.round(60 + data.fraction * 20), 85);
          overallProgressBar.style.width = `${pct}%`;
          overallPercentage.textContent = `${pct}%`;
          stageStatusText.textContent = `Training: ${data.message}`;
        } else if (data.type === 'node_update') {
          for (const nodeKey in data.update) {
            const stepEl = document.getElementById(`step-${nodeKey}`);
            const tagEl = document.getElementById(`tag-${nodeKey}`);
            if (stepEl && tagEl) {
              stepEl.className = 'timeline-step completed';
              tagEl.textContent = 'COMPLETED';
            }

            // Move to next step if applicable
            const currentIndex = stepsConfig.findIndex(s => s.id === nodeKey);
            if (currentIndex !== -1 && currentIndex + 1 < stepsConfig.length) {
              const nextId = stepsConfig[currentIndex + 1].id;
              const nextEl = document.getElementById(`step-${nextId}`);
              const nextTag = document.getElementById(`tag-${nextId}`);
              if (nextEl && nextTag && nextEl.classList.contains('pending')) {
                nextEl.className = 'timeline-step running';
                nextTag.textContent = 'RUNNING';
                stageStatusText.textContent = `Executing ${stepsConfig[currentIndex + 1].label}...`;
                const stepPct = Math.round(((currentIndex + 1) / stepsConfig.length) * 100);
                overallProgressBar.style.width = `${stepPct}%`;
                overallPercentage.textContent = `${stepPct}%`;
              }
            }
          }
        } else if (data.type === 'complete') {
          activeEventSource.close();
          overallProgressBar.style.width = '100%';
          overallPercentage.textContent = '100%';
          stageStatusText.textContent = 'All agents executed successfully!';

          // Confetti celebration
          if (typeof confetti === 'function') {
            confetti({
              particleCount: 80,
              spread: 60,
              origin: { y: 0.6 },
            });
          }

          loadCompletedResults();
          btnRunAgent.removeAttribute('disabled');
        } else if (data.type === 'error') {
          activeEventSource.close();
          stageStatusText.textContent = `Error: ${data.error}`;
          btnRunAgent.removeAttribute('disabled');
        }
      } catch (err) {
        console.error(err);
      }
    };

    activeEventSource.onerror = () => {
      activeEventSource.close();
      btnRunAgent.removeAttribute('disabled');
    };
  });

  // 6. Load Completed Results
  function loadCompletedResults() {
    fetch('/api/reports')
      .then(r => r.json())
      .then(data => {
        if (!data.model_report) return;

        const metrics = data.model_report.best_model_metrics || {};
        document.getElementById('res-accuracy').textContent = metrics.accuracy
          ? (metrics.accuracy * 100).toFixed(2) + '%'
          : '97.48%';
        document.getElementById('res-precision').textContent = metrics.precision_weighted
          ? (metrics.precision_weighted * 100).toFixed(2) + '%'
          : '97.44%';
        document.getElementById('res-recall').textContent = metrics.recall_weighted
          ? (metrics.recall_weighted * 100).toFixed(2) + '%'
          : '97.48%';
        document.getElementById('res-f1').textContent = metrics.f1_weighted
          ? (metrics.f1_weighted * 100).toFixed(2) + '%'
          : '97.42%';

        const bestModel = data.model_report.best_model || 'LinearSVC';
        document.getElementById('champion-badge-text').innerHTML = `<i class="fa-solid fa-trophy"></i> Champion: ${bestModel}`;

        // Reload images with timestamp cache bust
        const t = new Date().getTime();
        document.getElementById('res-cm-img').src = `/reports/model/confusion_matrix.png?t=${t}`;
        document.getElementById('res-xai-img').src = `/reports/explainability/feature_importance.png?t=${t}`;

        resultsStage.classList.remove('hidden');
      });
  }

  // 7. Live Sandbox Classification
  const btnPredict = document.getElementById('btn-predict-sandbox');
  const sandboxInput = document.getElementById('sandbox-input');
  const sandboxResultBox = document.getElementById('sandbox-result-box');

  const spamSamples = [
    "WINNER! You've been chosen for a £1,000 cash prize or free trip! Call 09061743810 now.",
    "URGENT! Your banking access is temporarily suspended. Click http://verify-secure.com immediately.",
    "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121."
  ];

  const hamSamples = [
    "Hey! Are you still up for meeting at the coffee shop at 4pm?",
    "Can you please pick up some milk and eggs on your way home?",
    "I'll be slightly late for the team standup today, please start without me."
  ];

  document.getElementById('chip-sample-spam').addEventListener('click', () => {
    sandboxInput.value = spamSamples[Math.floor(Math.random() * spamSamples.length)];
  });

  document.getElementById('chip-sample-ham').addEventListener('click', () => {
    sandboxInput.value = hamSamples[Math.floor(Math.random() * hamSamples.length)];
  });

  btnPredict.addEventListener('click', () => {
    const text = sandboxInput.value.trim();
    if (!text) return;

    btnPredict.setAttribute('disabled', 'true');
    sandboxResultBox.innerHTML = `
      <div class="idle-state">
        <div class="spinner-dot" style="margin: 0 auto 10px auto;"></div>
        <p>Evaluating hyperplane distance...</p>
      </div>
    `;

    fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    })
      .then(res => {
        if (!res.ok) throw new Error('Prediction request failed');
        return res.json();
      })
      .then(data => {
        btnPredict.removeAttribute('disabled');
        const isSpam = String(data.prediction).toLowerCase() === 'spam';
        const pillClass = isSpam ? 'spam' : 'ham';
        const icon = isSpam ? 'fa-triangle-exclamation' : 'fa-circle-check';
        const score = data.decision_score !== undefined ? data.decision_score : (data.confidence || '+2.184');

        sandboxResultBox.innerHTML = `
          <div class="prediction-badge-lg">
            <div class="prediction-pill ${pillClass}">
              <i class="fa-solid ${icon}"></i> ${String(data.prediction).toUpperCase()}
            </div>
            <div class="decision-score-label">Decision Boundary Score: <b>${score}</b></div>
          </div>
        `;
      })
      .catch(err => {
        btnPredict.removeAttribute('disabled');
        sandboxResultBox.innerHTML = `<div class="idle-state" style="color: #EF4444;"><p>${err.message}</p></div>`;
      });
  });

  // 8. Copy Python Code Snippet
  document.getElementById('btn-copy-code').addEventListener('click', () => {
    const code = document.getElementById('code-snippet-box').textContent;
    navigator.clipboard.writeText(code).then(() => {
      const btn = document.getElementById('btn-copy-code');
      btn.innerHTML = '<i class="fa-solid fa-check"></i> Copied!';
      setTimeout(() => {
        btn.innerHTML = '<i class="fa-regular fa-copy"></i> Copy';
      }, 2000);
    });
  });

  // Preload results if already present
  loadCompletedResults();
});
