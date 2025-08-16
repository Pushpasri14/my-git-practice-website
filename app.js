const form = document.getElementById('uploadForm');
const startBtn = document.getElementById('startBtn');
const statusPill = document.getElementById('statusPill');
const timeRange = document.getElementById('timeRange');
const yearEl = document.getElementById('year');
if (yearEl) yearEl.textContent = new Date().getFullYear();

const domEmotionEl = document.getElementById('dominantEmotion');
const emotionBarsEl = document.getElementById('emotionBars');
const gazeEl = document.getElementById('gaze');
const eyeTurnsEl = document.getElementById('eyeTurns');
const smileEl = document.getElementById('smile');
const logLinkEl = document.getElementById('logLink');
const videoLinkEl = document.getElementById('videoLink');
const previewVideo = document.getElementById('previewVideo');
const refreshShotsBtn = document.getElementById('refreshShots');
const screenshotsEl = document.getElementById('screenshots');

let sessionId = null;
let pollTimer = null;

function setPill(status) {
	if (!statusPill) return;
	statusPill.classList.remove('running','idle','error','done');
	statusPill.textContent = status;
	if (status === 'running') statusPill.classList.add('running');
	else if (status === 'idle') statusPill.classList.add('idle');
	else if (status === 'completed') statusPill.classList.add('done');
	else if (status === 'error') statusPill.classList.add('error');
}

form.addEventListener('submit', async (e) => {
	e.preventDefault();
	const fd = new FormData();
	fd.append('user', document.getElementById('user').value || 'User');
	const file = document.getElementById('video').files[0];
	if (!file) return;
	fd.append('file', file);
	startBtn.disabled = true;
	setPill('running');
	const res = await fetch('/api/analyze', { method: 'POST', body: fd });
	if (!res.ok) {
		setPill('error');
		startBtn.disabled = false;
		return;
	}
	const data = await res.json();
	sessionId = data.session_id;
	startPolling();
});

if (refreshShotsBtn) refreshShotsBtn.addEventListener('click', () => { if (sessionId) refreshScreenshots(''); });

function startPolling() {
	if (pollTimer) clearInterval(pollTimer);
	pollTimer = setInterval(async () => {
		if (!sessionId) return;
		const res = await fetch(`/api/status/${sessionId}`);
		if (res.status === 404) {
			clearInterval(pollTimer);
			setPill('idle');
			sessionId = null;
			return;
		}
		if (!res.ok) return;
		const data = await res.json();
		renderStatus(data);
		if (data.status === 'completed' || data.status === 'error') {
			clearInterval(pollTimer);
		}
		if (data.status === 'running') {
			refreshScreenshots('');
		}
	}, 1000);
}

async function renderStatus(data) {
	setPill(data.status);
	if (data.state) {
		const s = data.state;
		domEmotionEl.textContent = s.dominant_emotion || '-';
		gazeEl.textContent = s.current_gaze_direction ? s.current_gaze_direction.toUpperCase() : '-';
		eyeTurnsEl.textContent = `${s.eye_turn_count || 0}`;
		smileEl.textContent = `${(s.smile_confidence || 0).toFixed(2)}`;
		timeRange.textContent = `${formatTime(s.video_timestamp || 0)}`;
		renderEmotions(s.emotions || {});
	}
	if (data.outputs) {
		const { log_file, analyzed_video, screenshots_folder } = data.outputs;
		logLinkEl.innerHTML = log_file ? `<a href="/api/file?path=${encodeURIComponent(log_file)}" target="_blank">Download Log</a>` : '';
		videoLinkEl.innerHTML = analyzed_video ? `<a href="/api/file?path=${encodeURIComponent(analyzed_video)}&download=true" target="_blank">Download Analyzed Video</a>` : '';
		if (analyzed_video && previewVideo) {
			previewVideo.src = `/api/file?path=${encodeURIComponent(analyzed_video)}`;
		}
		if (screenshots_folder) {
			refreshScreenshots(screenshots_folder);
		} else if (sessionId) {
			try {
				const p = await fetch(`/api/session_paths/${sessionId}`);
				if (p.ok) {
					const j = await p.json();
					if (j.screenshots_folder) refreshScreenshots(j.screenshots_folder);
				}
			} catch {}
		}
	}
}

function renderEmotions(emotions) {
	emotionBarsEl.innerHTML = '';
	const entries = Object.entries(emotions).sort((a,b) => b[1]-a[1]).slice(0,7);
	for (const [name, score] of entries) {
		const row = document.createElement('div');
		row.className = 'bar-row';
		row.innerHTML = `
			<div class="bar-label">${capitalize(name)}</div>
			<div class="bar"><div class="bar-fill" style="width:${Math.min(100, score)}%"></div></div>
			<div class="bar-val">${score.toFixed(1)}%</div>
		`;
		emotionBarsEl.appendChild(row);
	}
}

async function refreshScreenshots(folder) {
	screenshotsEl.innerHTML = '';
	try {
		const res = await fetch(`/api/list_screenshots_by_session/${sessionId}`);
		if (!res.ok) throw new Error('list failed');
		const data = await res.json();
		for (const url of data.files) {
			const img = document.createElement('img');
			img.loading = 'lazy';
			img.src = url;
			screenshotsEl.appendChild(img);
		}
		if ((data.files || []).length === 0) {
			const hint = document.createElement('div');
			hint.className = 'bar-label';
			hint.textContent = 'No screenshots yet.';
			screenshotsEl.appendChild(hint);
		}
	} catch (e) {
		const hint = document.createElement('div');
		hint.className = 'bar-label';
		hint.textContent = 'Unable to load screenshots.';
		screenshotsEl.appendChild(hint);
	}
}

function formatTime(sec) {
	const m = Math.floor(sec / 60);
	const s = Math.floor(sec % 60);
	return `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
}
function capitalize(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : s; }