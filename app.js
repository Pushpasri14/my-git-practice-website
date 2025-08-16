const form = document.getElementById('uploadForm');
const sessionBlock = document.getElementById('session');
const statusEl = document.getElementById('status');
const progressEl = document.getElementById('progress');
const domEmotionEl = document.getElementById('dominantEmotion');
const emotionBarsEl = document.getElementById('emotionBars');
const gazeEl = document.getElementById('gaze');
const eyeTurnsEl = document.getElementById('eyeTurns');
const smileEl = document.getElementById('smile');
const logLinkEl = document.getElementById('logLink');
const summaryLinkEl = document.getElementById('summaryLink');
const videoLinkEl = document.getElementById('videoLink');
const screenshotsEl = document.getElementById('screenshots');

let sessionId = null;
let pollTimer = null;

form.addEventListener('submit', async (e) => {
	e.preventDefault();
	const fd = new FormData();
	fd.append('user', document.getElementById('user').value || 'User');
	const file = document.getElementById('video').files[0];
	if (!file) return;
	fd.append('file', file);

	statusEl.textContent = 'uploading...';
	sessionBlock.classList.remove('hidden');

	const res = await fetch('/api/analyze', { method: 'POST', body: fd });
	if (!res.ok) {
		statusEl.textContent = 'failed to start analysis';
		return;
	}
	const data = await res.json();
	sessionId = data.session_id;
	statusEl.textContent = 'running';
	startPolling();
});

function startPolling() {
	if (pollTimer) clearInterval(pollTimer);
	pollTimer = setInterval(async () => {
		if (!sessionId) return;
		const res = await fetch(`/api/status/${sessionId}`);
		if (!res.ok) return;
		const data = await res.json();
		renderStatus(data);
		if (data.status === 'completed' || data.status === 'error') {
			clearInterval(pollTimer);
		}
	}, 1000);
}

async function renderStatus(data) {
	statusEl.textContent = data.status;
	if (data.state) {
		const s = data.state;
		domEmotionEl.textContent = s.dominant_emotion || '-';
		gazeEl.textContent = `Gaze: ${s.current_gaze_direction || '-'}  |  ${formatTime(s.video_timestamp || 0)}`;
		eyeTurnsEl.textContent = `Eye Turns: ${s.eye_turn_count || 0}`;
		smileEl.textContent = `Smile: ${(s.smile_confidence || 0).toFixed(2)}`;
		renderEmotions(s.emotions || {});
	}
	if (data.outputs) {
		const { log_file, analyzed_video, screenshots_folder } = data.outputs;
		logLinkEl.innerHTML = log_file ? `<a href="/api/file?path=${encodeURIComponent(log_file)}" target="_blank">Download Log</a>` : '';
		videoLinkEl.innerHTML = analyzed_video ? `<a href="/api/file?path=${encodeURIComponent(analyzed_video)}&download=true" target="_blank">Download Analyzed Video</a>` : '';
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
		row.innerHTML = `<div style="display:flex; justify-content:space-between; font-size:12px; color:#94a3b8;">
			<span>${capitalize(name)}</span><span>${score.toFixed(1)}%</span>
		</div>
		<div class="bar"><div style="width:${Math.min(100, score)}%"></div></div>`;
		emotionBarsEl.appendChild(row);
	}
}

async function refreshScreenshots(folder) {
	screenshotsEl.innerHTML = '';
	try {
		const res = await fetch(`/api/list_screenshots?path=${encodeURIComponent(folder)}`);
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
			hint.style.color = '#94a3b8';
			hint.textContent = 'No screenshots yet. Hold gaze left/right for 2s.';
			screenshotsEl.appendChild(hint);
		}
	} catch (e) {
		const hint = document.createElement('div');
		hint.style.color = '#94a3b8';
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