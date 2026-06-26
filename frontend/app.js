document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const apiUrlInput = document.getElementById('api-url');
    const mlflowUrlInput = document.getElementById('mlflow-url');
    const ollamaUrlInput = document.getElementById('ollama-url');

    const projectSelect = document.getElementById('project-select');
    const projectDesc = document.getElementById('project-desc');
    const newProjectBtn = document.getElementById('toggle-new-project-btn');
    const newProjectForm = document.getElementById('new-project-form');
    const createProjectBtn = document.getElementById('create-project-btn');
    const newProjNameInput = document.getElementById('new-proj-name');
    const newProjDescInput = document.getElementById('new-proj-desc');

    const datasetSelect = document.getElementById('dataset-select');
    const datasetInfo = document.getElementById('dataset-info');
    const uploadBtn = document.getElementById('toggle-upload-btn');
    const uploadForm = document.getElementById('upload-form');
    const doUploadBtn = document.getElementById('upload-btn');
    const uploadDsName = document.getElementById('upload-ds-name');
    const uploadFile = document.getElementById('upload-file');
    const uploadProgressContainer = document.getElementById('upload-progress-container');
    const uploadProgress = document.getElementById('upload-progress');
    const uploadStatusText = document.getElementById('upload-status-text');

    const expNameInput = document.getElementById('exp-name');
    const targetAccInput = document.getElementById('target-acc');
    const targetAccVal = document.getElementById('target-acc-val');
    const maxCyclesInput = document.getElementById('max-cycles');
    const maxCyclesVal = document.getElementById('max-cycles-val');
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const startWarning = document.getElementById('start-warning');

    const tabs = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    // State Variables
    let datasetsData = [];
    let projectsData = [];
    let selectedDatasetPath = null;
    let selectedDatasetYaml = null;
    let agentPollingInterval = null;
    let mlflowIframe = document.getElementById('mlflow-iframe');
    let isAgentRunning = false;

    // --- Tab Switching ---
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById(tab.dataset.tab).classList.add('active');
            
            if (tab.dataset.tab === 'tab-mlflow') {
                mlflowIframe.src = mlflowUrlInput.value;
            } else if (tab.dataset.tab === 'tab-data') {
                fetchDatasets();
            } else if (tab.dataset.tab === 'tab-history') {
                fetchJobs();
            }
        });
    });

    // --- Collapsible Sections ---
    document.querySelectorAll('.section-header').forEach(header => {
        header.addEventListener('click', () => {
            const content = header.nextElementSibling;
            const icon = header.querySelector('.toggle-icon');
            content.classList.toggle('hidden');
            icon.textContent = content.classList.contains('hidden') ? '▼' : '▲';
        });
    });

    // --- Sliders ---
    targetAccInput.addEventListener('input', (e) => targetAccVal.textContent = e.target.value);
    maxCyclesInput.addEventListener('input', (e) => maxCyclesVal.textContent = e.target.value);

    // --- Toggle Forms ---
    newProjectBtn.addEventListener('click', () => {
        newProjectForm.classList.toggle('hidden');
    });

    uploadBtn.addEventListener('click', () => {
        uploadForm.classList.toggle('hidden');
    });

    // --- Check Readiness ---
    function checkReadiness() {
        if (projectSelect.value && datasetSelect.value && !isAgentRunning) {
            startBtn.disabled = false;
            startWarning.classList.add('hidden');
        } else {
            startBtn.disabled = true;
            if (!isAgentRunning) startWarning.classList.remove('hidden');
        }
    }

    // --- Helper for URLs ---
    function getApiUrl() {
        let url = apiUrlInput.value.trim();
        if (!url.startsWith('http://') && !url.startsWith('https://')) {
            url = 'http://' + url;
        }
        if (url.endsWith('/')) {
            url = url.slice(0, -1);
        }
        return url;
    }

    // --- Fetch API Data ---
    async function fetchProjects() {
        try {
            const res = await fetch(`${getApiUrl()}/api/projects`);
            projectsData = await res.json();
            projectSelect.innerHTML = '<option value="">Select Project</option>';
            projectsData.forEach(p => {
                const opt = document.createElement('option');
                opt.value = p.name;
                opt.textContent = p.name;
                projectSelect.appendChild(opt);
            });
            projectSelect.addEventListener('change', () => {
                const p = projectsData.find(x => x.name === projectSelect.value);
                projectDesc.textContent = p && p.description ? `📝 ${p.description}` : '';
                checkReadiness();
            });
        } catch (e) {
            console.error('Failed to fetch projects', e);
        }
    }

    async function fetchDatasets() {
        try {
            const res = await fetch(`${getApiUrl()}/api/data/list`);
            datasetsData = await res.json();
            
            // Update Sidebar
            datasetSelect.innerHTML = '<option value="">Select Dataset</option>';
            datasetsData.forEach(d => {
                const opt = document.createElement('option');
                opt.value = d.dataset_name;
                opt.textContent = d.dataset_name;
                datasetSelect.appendChild(opt);
            });
            
            datasetSelect.addEventListener('change', () => {
                const d = datasetsData.find(x => x.dataset_name === datasetSelect.value);
                if (d) {
                    datasetInfo.textContent = `📂 ${d.size_mb} MB | 🖼 ${d.image_count} images`;
                    selectedDatasetPath = d.server_path;
                    selectedDatasetYaml = d.yaml_path;
                } else {
                    datasetInfo.textContent = '';
                    selectedDatasetPath = null;
                    selectedDatasetYaml = null;
                }
                checkReadiness();
            });

            // Update Data Management Tab
            const dataListContainer = document.getElementById('data-list-container');
            if(dataListContainer) {
                if (datasetsData.length === 0) {
                    dataListContainer.innerHTML = '<p class="info-box">No datasets uploaded yet.</p>';
                } else {
                    let html = '';
                    datasetsData.forEach(ds => {
                        html += `
                            <div style="display:flex; justify-content:space-between; align-items:center; background: rgba(0,0,0,0.2); padding: 16px; border-radius: 8px; margin-bottom: 12px; border: 1px solid rgba(255,255,255,0.1);">
                                <div>
                                    <h4 style="color:var(--primary); margin-bottom:4px;">${ds.dataset_name}</h4>
                                    <p class="caption">Path: ${ds.server_path}</p>
                                    ${ds.yaml_path ? `<p class="caption">YAML: ${ds.yaml_path}</p>` : ''}
                                </div>
                                <div style="text-align: right;">
                                    <p><strong>${ds.size_mb} MB</strong> | ${ds.image_count} images</p>
                                    <button class="danger-btn" style="margin-top:8px; padding: 6px 12px;" onclick="deleteDataset('${ds.dataset_name}')">🗑️ Delete</button>
                                </div>
                            </div>
                        `;
                    });
                    dataListContainer.innerHTML = html;
                }
            }
        } catch (e) {
            console.error('Failed to fetch datasets', e);
        }
    }

    window.deleteDataset = async function(name) {
        if(confirm(`Are you sure you want to delete dataset ${name}?`)) {
            try {
                await fetch(`${getApiUrl()}/api/data/${name}`, { method: 'DELETE' });
                fetchDatasets();
            } catch(e) {
                alert('Error deleting dataset');
            }
        }
    };

    // --- Create Project ---
    createProjectBtn.addEventListener('click', async () => {
        const name = newProjNameInput.value.trim();
        const desc = newProjDescInput.value.trim();
        if (!name) return alert('Enter a project name');
        
        try {
            const res = await fetch(`${getApiUrl()}/api/projects`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, description: desc })
            });
            if (res.ok) {
                newProjNameInput.value = '';
                newProjDescInput.value = '';
                newProjectForm.classList.add('hidden');
                await fetchProjects();
                projectSelect.value = name;
                projectSelect.dispatchEvent(new Event('change'));
            } else {
                const err = await res.json();
                alert(`Error: ${err.detail}`);
            }
        } catch (e) {
            alert('Failed to create project: ' + e.message);
        }
    });

    // --- Chunked Upload ---
    doUploadBtn.addEventListener('click', async () => {
        const file = uploadFile.files[0];
        const dsName = uploadDsName.value.trim();
        if (!file || !dsName) return alert('Select a file and enter dataset name');
        
        uploadProgressContainer.classList.remove('hidden');
        doUploadBtn.disabled = true;

        try {
            // Init
            let initRes = await fetch(`${getApiUrl()}/api/data/upload/init`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dataset_name: dsName, filename: file.name })
            });
            const { upload_id } = await initRes.json();

            // Chunks
            const CHUNK_SIZE = 2 * 1024 * 1024;
            const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
            
            for (let i = 0; i < totalChunks; i++) {
                const start = i * CHUNK_SIZE;
                const end = Math.min(file.size, start + CHUNK_SIZE);
                const chunk = file.slice(start, end);

                const formData = new FormData();
                formData.append('chunk_index', i);
                formData.append('chunk', chunk, `chunk_${i}`);

                await fetch(`${getApiUrl()}/api/data/upload/chunk/${upload_id}`, {
                    method: 'POST',
                    body: formData
                });

                const pct = Math.round(((i + 1) / totalChunks) * 100);
                uploadProgress.style.width = `${pct}%`;
                uploadStatusText.textContent = `${pct}%`;
            }

            // Finalize
            await fetch(`${getApiUrl()}/api/data/upload/finalize/${upload_id}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ total_chunks: totalChunks })
            });

            uploadStatusText.textContent = 'Complete!';
            setTimeout(() => {
                uploadProgressContainer.classList.add('hidden');
                uploadForm.classList.add('hidden');
                doUploadBtn.disabled = false;
                uploadProgress.style.width = '0%';
                uploadFile.value = '';
                uploadDsName.value = '';
                fetchDatasets();
            }, 1000);

        } catch (e) {
            alert('Upload failed: ' + e.message);
            doUploadBtn.disabled = false;
        }
    });

    // --- Fetch Jobs & History Tab ---
    document.getElementById('refresh-jobs-btn').addEventListener('click', fetchJobs);
    document.getElementById('refresh-data-btn').addEventListener('click', fetchDatasets);

    async function fetchJobs() {
        try {
            const res = await fetch(`${getApiUrl()}/api/jobs`);
            const jobs = await res.json();
            const tbody = document.querySelector('#jobs-table tbody');
            const dlSelect = document.getElementById('download-run-select');
            
            tbody.innerHTML = '';
            dlSelect.innerHTML = '<option value="">Select run</option>';
            
            jobs.forEach(j => {
                let mAP50 = 0, precision = 0, recall = 0;
                try {
                    if (j.metrics) {
                        const m = JSON.parse(j.metrics);
                        mAP50 = (m.map50 || 0).toFixed(4);
                        precision = (m.precision || 0).toFixed(4);
                        recall = (m.recall || 0).toFixed(4);
                    }
                } catch(e) {}

                // Table row
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${j.run_id}</td>
                    <td>${j.project_name}</td>
                    <td>${j.experiment_name}</td>
                    <td>${j.model_scale}</td>
                    <td><span class="${j.status === 'completed' ? 'badge-done' : (j.status === 'running' ? 'badge-running' : 'badge-fail')}">${j.status.toUpperCase()}</span></td>
                    <td style="color:var(--primary); font-weight:bold;">${mAP50}</td>
                    <td>${precision}</td>
                    <td>${recall}</td>
                `;
                tbody.appendChild(tr);

                // Download select
                const opt = document.createElement('option');
                opt.value = j.run_id;
                opt.textContent = `${j.project_name} - ${j.experiment_name} [${j.status}]`;
                dlSelect.appendChild(opt);
            });

            dlSelect.addEventListener('change', () => {
                const dlLink = document.getElementById('download-link');
                if (dlSelect.value) {
                    dlLink.href = `${getApiUrl()}/api/train/download/${dlSelect.value}`;
                    dlLink.removeAttribute('disabled');
                } else {
                    dlLink.href = '#';
                    dlLink.setAttribute('disabled', 'true');
                }
            });

        } catch (e) {
            console.error('Failed to fetch jobs', e);
        }
    }

    // --- Start Agent ---
    startBtn.addEventListener('click', async () => {
        const payload = {
            project_name: projectSelect.value,
            project_description: projectsData.find(x => x.name === projectSelect.value)?.description || "",
            experiment_name: expNameInput.value.trim(),
            dataset_path: selectedDatasetYaml || selectedDatasetPath,
            target_accuracy: parseFloat(targetAccInput.value),
            max_cycles: parseInt(maxCyclesInput.value),
            api_url: getApiUrl(),
            mlflow_uri: mlflowUrlInput.value,
            ollama_url: ollamaUrlInput.value
        };

        try {
            const res = await fetch('/api/agent/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            if (res.ok) {
                startPolling();
                document.querySelector('[data-tab="tab-loop"]').click();
            } else {
                const err = await res.json();
                alert(`Cannot start agent: ${err.detail}`);
            }
        } catch (e) {
            alert('Cannot reach orchestrator API. Ensure orchestrator.py is running.');
        }
    });

    stopBtn.addEventListener('click', async () => {
        try {
            await fetch('/api/agent/stop', { method: 'DELETE' });
        } catch (e) {}
    });

    // --- Poll Agent State ---
    function startPolling() {
        if (agentPollingInterval) clearInterval(agentPollingInterval);
        document.getElementById('loop-empty-state').classList.add('hidden');
        document.getElementById('loop-active-state').classList.remove('hidden');
        
        agentPollingInterval = setInterval(async () => {
            try {
                const res = await fetch('/api/agent/state');
                const data = await res.json();
                updateLoopUI(data);
                
                if (!data.is_running && data.state.status && data.state.status !== 'Starting…') {
                    // One final update then stop polling
                    isAgentRunning = false;
                    clearInterval(agentPollingInterval);
                    checkReadiness();
                    startBtn.classList.remove('hidden');
                    stopBtn.classList.add('hidden');
                } else {
                    isAgentRunning = true;
                    startBtn.classList.add('hidden');
                    stopBtn.classList.remove('hidden');
                    checkReadiness();
                }
            } catch (e) {
                console.error("Polling error", e);
            }
        }, 2000);
    }

    // Initialize polling just in case it's already running on reload
    startPolling();

    // --- Update Loop UI ---
    function updateLoopUI(data) {
        const state = data.state;
        if (!state || Object.keys(state).length === 0) return;

        // Status Alert
        const alertBox = document.getElementById('agent-status-alert');
        if (state.error) {
            alertBox.className = 'alert error';
            alertBox.innerHTML = `❌ <strong>Error:</strong> ${state.error}`;
        } else if (data.is_running) {
            alertBox.className = 'alert info';
            alertBox.innerHTML = `🔄 <strong>Status:</strong> ${state.status || 'Running...'}`;
        } else {
            alertBox.className = 'alert success';
            alertBox.innerHTML = `✅ <strong>Status:</strong> ${state.status || 'Completed.'}`;
        }

        // Stepper logic
        const s = state.status || '';
        const cycle = state.current_cycle || 0;
        const histLen = (state.history || []).length;
        
        const setStep = (id, isActive, isDone) => {
            const step = document.getElementById(`step-${id}`);
            step.classList.remove('active', 'done');
            if (isDone) step.classList.add('done');
            else if (isActive) step.classList.add('active');
        };

        const hasStats = state.data_stats && Object.keys(state.data_stats).length > 0;
        const hasModel = !!state.current_model;

        setStep(1, s.toLowerCase().includes('analyz'), hasStats);
        setStep(2, s.toLowerCase().includes('select'), hasModel);
        setStep(3, s.toLowerCase().includes('train'), histLen >= cycle && cycle > 0);
        setStep(4, s.toLowerCase().includes('evaluat'), histLen >= cycle && cycle > 0);
        setStep(5, s.toLowerCase().includes('adjust') || s.toLowerCase().includes('tuning'), cycle > 1 && histLen > 0);

        // Metrics
        let bestMap = 0;
        (state.history || []).forEach(h => {
            const map = h.metrics?.map50 || 0;
            if (map > bestMap) bestMap = map;
        });

        document.getElementById('metric-cycle').textContent = `${cycle} / ${state.max_cycles || '?'}`;
        document.getElementById('metric-model').textContent = state.current_model ? `${state.current_model}.pt` : '—';
        document.getElementById('metric-map50').textContent = bestMap.toFixed(4);
        document.getElementById('metric-target').textContent = state.target_accuracy || '0.80';

        // Dataset Stats
        if (hasStats) {
            document.getElementById('dataset-stats-container').classList.remove('hidden');
            const st = state.data_stats;
            document.getElementById('stat-classes').textContent = st.num_classes;
            document.getElementById('stat-names').textContent = (st.class_names || []).join(', ');
            document.getElementById('stat-images').textContent = st.total_images;
            document.getElementById('stat-boxes').textContent = st.total_boxes;
            document.getElementById('stat-small').textContent = st.bbox_distribution?.small || 0;
            document.getElementById('stat-medium').textContent = st.bbox_distribution?.medium || 0;
            document.getElementById('stat-large').textContent = st.bbox_distribution?.large || 0;
        }

        // Live Monitor
        if (state.run_id && (data.is_running || s.startsWith('Cycle'))) {
            document.getElementById('live-monitor-container').classList.remove('hidden');
            // Fetch live progress from GPU API directly
            fetch(`${getApiUrl()}/api/train/status/${state.run_id}`)
                .then(r => r.json())
                .then(live => {
                    const prog = live.progress || {};
                    const pct = prog.pct || 0;
                    document.getElementById('train-progress').style.width = `${pct}%`;
                    document.getElementById('train-progress-text').textContent = `Epoch ${prog.current_epoch}/${prog.total_epochs} — ${pct}%`;
                    
                    const logsBox = document.getElementById('train-logs');
                    if (live.logs) {
                        logsBox.textContent = live.logs;
                        logsBox.scrollTop = logsBox.scrollHeight;
                    }
                }).catch(e => {});
        } else {
            document.getElementById('live-monitor-container').classList.add('hidden');
        }

        // Cycle History
        const histContainer = document.getElementById('history-container');
        const histList = document.getElementById('history-list');
        if (state.history && state.history.length > 0) {
            histContainer.classList.remove('hidden');
            let hHtml = '';
            [...state.history].reverse().forEach(h => {
                const m = h.metrics || {};
                const hp = h.hyperparams || {};
                hHtml += `
                    <div style="background:var(--bg-panel); border:1px solid var(--border); border-radius:8px; margin-bottom:16px; overflow:hidden;">
                        <div style="padding:16px; background:rgba(0,0,0,0.2); border-bottom:1px solid var(--border); display:flex; justify-content:space-between;">
                            <strong style="color:var(--primary);">Cycle ${h.cycle} — ${h.model}.pt</strong>
                            <strong style="color:var(--accent-success);">mAP50: ${(m.map50 || 0).toFixed(4)}</strong>
                        </div>
                        <div style="padding:16px;">
                            <div class="alert info" style="margin-bottom:16px; padding:12px;"><strong>LLM Reasoning:</strong> ${h.reasoning || '—'}</div>
                            <div style="display:flex; gap:24px;">
                                <div style="flex:1;">
                                    <h4 style="margin-bottom:8px; color:var(--text-muted);">Metrics</h4>
                                    <ul style="list-style:none; font-family:monospace; color:var(--text-main);">
                                        <li>Precision: ${(m.precision || 0).toFixed(4)}</li>
                                        <li>Recall: ${(m.recall || 0).toFixed(4)}</li>
                                        <li>Val Box Loss: ${(m.val_box_loss || 0).toFixed(4)}</li>
                                    </ul>
                                </div>
                                <div style="flex:1;">
                                    <h4 style="margin-bottom:8px; color:var(--text-muted);">Hyperparameters</h4>
                                    <ul style="list-style:none; font-family:monospace; color:var(--text-main); display:grid; grid-template-columns: 1fr 1fr;">
                                        <li>epochs: ${hp.epochs}</li>
                                        <li>batch: ${hp.batch}</li>
                                        <li>imgsz: ${hp.imgsz}</li>
                                        <li>lr0: ${hp.lr0}</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            });
            histList.innerHTML = hHtml;
        }
    }

    // Initial Fetches
    fetchProjects();
    fetchDatasets();
});
