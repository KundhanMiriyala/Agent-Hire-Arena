const API_URL = window.location.origin;

// DOM Elements
const els = {
    budgetVal: document.getElementById('budget-val'),
    budgetBar: document.getElementById('budget-bar'),
    hiresVal: document.getElementById('hires-val'),
    hiresBar: document.getElementById('hires-bar'),
    scoreVal: document.getElementById('score-val'),
    scoreBar: document.getElementById('score-bar'),
    candidatesBody: document.getElementById('candidates-body'),
    taskSelect: document.getElementById('task-select'),
    deployBtn: document.getElementById('deploy-btn'),
    resetBtn: document.getElementById('reset-btn'),
    auditLog: document.getElementById('audit-log'),
    envStatus: document.getElementById('env-status'),
    modelStatus: document.getElementById('model-status'),
    modelSelect: document.getElementById('model-select'),
    getStateBtn: document.getElementById('get-state-btn'),
    manualAction: document.getElementById('manual-action'),
    manualCandidate: document.getElementById('manual-candidate'),
    manualStepBtn: document.getElementById('manual-step-btn'),
    agentStepBtn: document.getElementById('agent-step-btn'),
};

let currentTask = null;
let isDeploying = false;
let maxBudget = 100;

// Initialize
async function init() {
    await loadTasks();
    await resetEnv();

    els.resetBtn.addEventListener('click', resetEnv);
    els.deployBtn.addEventListener('click', toggleDeployment);
    els.getStateBtn.addEventListener('click', getState);
    els.manualStepBtn.addEventListener('click', doManualStep);
    els.agentStepBtn.addEventListener('click', doAgentStepSingle);
}

// Logging
function addLog(message, type = 'system') {
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.textContent = message;
    els.auditLog.appendChild(entry);
    els.auditLog.scrollTop = els.auditLog.scrollHeight;
}

async function getState() {
    try {
        const res = await fetch(`${API_URL}/state`);
        const state = await res.json();
        addLog(`Raw State: ${JSON.stringify(state, null, 2)}`, 'system');
    } catch (e) {
        addLog(`Error fetching state: ${e.message}`, 'error');
    }
}

async function doManualStep() {
    if (isDeploying) return;
    const action = els.manualAction.value;
    const candidate_id = els.manualCandidate.value;
    
    if (action !== 'finalize' && !candidate_id) {
        addLog('Select a candidate ID first', 'error');
        return;
    }
    
    try {
        const res = await fetch(`${API_URL}/step`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: action, candidate_id: candidate_id })
        });
        
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Step failed');
        }
        
        const data = await res.json();
        addLog(`Manual Action: {"action": "${action}", "candidate_id": "${candidate_id}"}`, 'action');
        
        updateUI(data.observation);
        
        if (data.observation.last_action_result) {
            addLog(`Result: ${data.observation.last_action_result}`, 'result');
        }
        if (data.reward && data.reward.step_reward !== undefined) {
             addLog(`Reward: ${data.reward.step_reward.toFixed(2)} (${data.reward.reason})`, 'system');
        }
        if (data.observation.done) {
            addLog('Episode completed. Waiting for final grader score...', 'system');
            await fetchMetrics();
        }
    } catch (e) {
        addLog(`Manual step error: ${e.message}`, 'error');
    }
}

async function doAgentStepSingle() {
    if (isDeploying) return;
    try {
        const selectedModelName = els.modelSelect.options[els.modelSelect.selectedIndex].text;
        addLog(`Running single step with ${selectedModelName}...`, 'system');
        
        const res = await fetch(`${API_URL}/agent_step`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: els.modelSelect.value })
        });
        
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Step failed');
        }
        
        const data = await res.json();
        
        if (data.fallback) {
            addLog(`⚠️ LLM unavailable — fell back to Heuristic Policy. Reason: ${data.fallback_reason}`, 'error');
        }
        addLog(`Reasoning: ${data.reasoning}`, 'reasoning');
        addLog(`Action: ${JSON.stringify(data.action)}`, 'action');
        updateUI(data.observation);
        
        if (data.observation.last_action_result) {
            addLog(`Result: ${data.observation.last_action_result}`, 'result');
        }
        if (data.reward && data.reward.step_reward !== undefined) {
             addLog(`Reward: ${data.reward.step_reward.toFixed(2)} (${data.reward.reason})`, 'system');
        }
        if (data.observation.done) {
            addLog('Episode completed. Waiting for final grader score...', 'system');
            await fetchMetrics();
        }
    } catch (e) {
        addLog(`Agent execution error: ${e.message}`, 'error');
    }
}

// API Calls
async function loadTasks() {
    try {
        const res = await fetch(`${API_URL}/tasks`);
        const tasks = await res.json();
        
        els.taskSelect.innerHTML = '';
        for (const [id, cfg] of Object.entries(tasks)) {
            const option = document.createElement('option');
            option.value = id;
            option.textContent = cfg.name || id;
            if (id === 'nightmare') option.selected = true;
            els.taskSelect.appendChild(option);
        }
    } catch (e) {
        addLog(`Error loading tasks: ${e.message}`, 'error');
    }
}

async function resetEnv() {
    if (isDeploying) return;
    
    const task = els.taskSelect.value;
    els.auditLog.innerHTML = '';
    addLog(`Initializing environment for task: ${task.toUpperCase()}`);
    
    els.envStatus.textContent = 'Resetting...';
    els.envStatus.className = 'status-badge';
    
    try {
        const res = await fetch(`${API_URL}/reset`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task: task })
        });
        const obs = await res.json();
        maxBudget = obs.budget_total || obs.budget_remaining; // Fallback if budget_total isn't exposed
        updateUI(obs);
        
        els.envStatus.textContent = 'Ready';
        els.envStatus.className = 'status-badge active';
        addLog('Environment ready. Awaiting deployment.', 'system');
    } catch (e) {
        addLog(`Reset failed: ${e.message}`, 'error');
        els.envStatus.textContent = 'Error';
        els.envStatus.className = 'status-badge';
    }
}

async function toggleDeployment() {
    if (isDeploying) {
        // We could implement stopping, but for now we just let it run or disable button
        return;
    }
    
    isDeploying = true;
    els.deployBtn.disabled = true;
    els.deployBtn.textContent = 'Deploying...';
    els.modelStatus.textContent = 'Running';
    els.modelStatus.className = 'status-badge running';
    
    const selectedModelName = els.modelSelect.options[els.modelSelect.selectedIndex].text;
    addLog(`Booting ${selectedModelName}...`, 'system');
    
    // Start polling the agent endpoint
    runAgentLoop();
}

async function runAgentLoop() {
    if (!isDeploying) return;
    
    try {
        const res = await fetch(`${API_URL}/agent_step`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: els.modelSelect.value })
        });
        
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Step failed');
        }
        
        const data = await res.json();
        
        // Notify if LLM failed and we fell back to heuristic
        if (data.fallback) {
            addLog(`⚠️ LLM unavailable — fell back to Heuristic Policy. Reason: ${data.fallback_reason}`, 'error');
        }
        
        // Log the thought process
        addLog(`Reasoning: ${data.reasoning}`, 'reasoning');
        addLog(`Action: ${JSON.stringify(data.action)}`, 'action');
        
        // Update state
        updateUI(data.observation);
        
        if (data.observation.last_action_result) {
            addLog(`Result: ${data.observation.last_action_result}`, 'result');
        }
        
        if (data.reward && data.reward.step_reward !== undefined) {
             addLog(`Reward: ${data.reward.step_reward.toFixed(2)} (${data.reward.reason})`, 'system');
        }
        
        if (data.observation.done) {
            isDeploying = false;
            els.deployBtn.disabled = false;
            els.deployBtn.textContent = 'Deploy';
            els.modelStatus.textContent = 'Finished';
            els.modelStatus.className = 'status-badge';
            addLog('Episode completed. Waiting for final grader score...', 'system');
            await fetchMetrics();
            return;
        }
        
        // Continue loop with slight delay for visualization
        setTimeout(runAgentLoop, 800);
        
    } catch (e) {
        addLog(`Agent execution error: ${e.message}`, 'error');
        isDeploying = false;
        els.deployBtn.disabled = false;
        els.deployBtn.textContent = 'Deploy';
        els.modelStatus.textContent = 'Error';
        els.modelStatus.className = 'status-badge';
    }
}

async function fetchMetrics() {
    try {
        const res = await fetch(`${API_URL}/metrics`);
        if (!res.ok) return;
        const data = await res.json();
        
        if (data.metrics && data.metrics.final_score !== undefined) {
            const score = data.metrics.final_score;
            els.scoreVal.textContent = score.toFixed(2);
            els.scoreBar.style.width = `${Math.min(100, Math.max(0, score * 100))}%`;
            
            // Color based on score
            if (score > 0.7) els.scoreBar.className = 'metric-bar green';
            else if (score > 0.4) els.scoreBar.className = 'metric-bar orange';
            else els.scoreBar.className = 'metric-bar red';
            
            addLog(`Final Grader Score: ${score.toFixed(2)}`, 'result');
        }
    } catch (e) {
        console.error("Could not fetch final metrics", e);
    }
}

function updateUI(obs) {
    // Metrics
    els.budgetVal.textContent = obs.budget_remaining.toFixed(0);
    const budgetPct = Math.max(0, (obs.budget_remaining / maxBudget) * 100);
    els.budgetBar.style.width = `${budgetPct}%`;
    
    if (budgetPct < 20) els.budgetBar.className = 'metric-bar red';
    else els.budgetBar.className = 'metric-bar orange';

    const hiresMade = obs.hires_made || [];
    els.hiresVal.textContent = hiresMade.length;
    els.hiresBar.style.width = `${Math.min(100, hiresMade.length * 20)}%`; // Assumes target is ~5
    
    // Reset score if not done
    if (!obs.done) {
        els.scoreVal.textContent = '---';
        els.scoreBar.style.width = '0%';
    }

    // Table
    els.candidatesBody.innerHTML = '';
    for (const c of obs.candidates) {
        const tr = document.createElement('tr');
        
        let statusHtml = '-';
        if (hiresMade.includes(c.candidate_id)) {
            statusHtml = '<span class="status-hired">Hired</span>';
        } else if ((obs.skipped || []).includes(c.candidate_id)) {
            statusHtml = '<span class="status-skipped">Skipped</span>';
        }
        
        let interviewScore = '-';
        if (obs.interviews_done && c.candidate_id in obs.interviews_done) {
            const sc = obs.interviews_done[c.candidate_id];
            const cls = sc > 0.7 ? 'val-high' : sc < 0.4 ? 'val-low' : 'val-med';
            interviewScore = `<span class="${cls}">${sc.toFixed(3)}</span>`;
        }
        
        let probeScore = '-';
        if (obs.probes_done && c.candidate_id in obs.probes_done) {
            const psc = obs.probes_done[c.candidate_id];
            const cls = psc > 0.7 ? 'val-high' : psc < 0.4 ? 'val-low' : 'val-med';
            probeScore = `<span class="${cls}">${psc.toFixed(3)}</span>`;
        }

        const resScore = c.resume_score;
        const resCls = resScore > 0.7 ? 'val-high' : resScore < 0.4 ? 'val-low' : 'val-med';

        tr.innerHTML = `
            <td>${c.candidate_id}</td>
            <td>${c.name} <br><small style="color:var(--text-secondary)">${c.role || ''} (${c.years_experience}y)</small></td>
            <td class="${resCls}">${resScore.toFixed(3)}</td>
            <td>${interviewScore}</td>
            <td>${probeScore}</td>
            <td>${statusHtml}</td>
        `;
        els.candidatesBody.appendChild(tr);
    }
    
    // Update manual candidate dropdown
    els.manualCandidate.innerHTML = '<option value="">-- None --</option>';
    for (const c of obs.candidates) {
        if (!hiresMade.includes(c.candidate_id) && !(obs.skipped || []).includes(c.candidate_id)) {
            const option = document.createElement('option');
            option.value = c.candidate_id;
            option.textContent = `${c.candidate_id} (${c.name})`;
            els.manualCandidate.appendChild(option);
        }
    }
}

// Start
init();
