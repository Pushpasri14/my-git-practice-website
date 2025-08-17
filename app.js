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
const themeToggle = document.getElementById('themeToggle');
const errorBox = document.getElementById('errorBox');

let sessionId = null;
let pollTimer = null;

// Minimal debug flag
let __loggedFirstStatus = false;

(function initTheme() {
	const root = document.documentElement;
	const media = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)');
	const stored = localStorage.getItem('theme');
	function applyTheme(theme) {
		if (theme === 'light') {
			root.setAttribute('data-theme', 'light');
			if (themeToggle) { themeToggle.textContent = 'Dark'; themeToggle.setAttribute('aria-pressed', 'true'); }
		} else {
			root.removeAttribute('data-theme');
			if (themeToggle) { themeToggle.textContent = 'Light'; themeToggle.setAttribute('aria-pressed', 'false'); }
		}
		localStorage.setItem('theme', theme);
	}
	applyTheme(stored || (media && media.matches ? 'dark' : 'dark'));
	if (themeToggle) {
		themeToggle.addEventListener('click', () => {
			const isLight = root.getAttribute('data-theme') === 'light';
			applyTheme(isLight ? 'dark' : 'light');
		});
	}
	if (media && media.addEventListener) {
		media.addEventListener('change', (e) => {
			if (!localStorage.getItem('theme')) {
				applyTheme(e.matches ? 'dark' : 'light');
			}
		});
	}
})();

function setPill(status) {
	if (!statusPill) return;
	statusPill.classList.remove('running','idle','error','done');
	statusPill.textContent = status;
	if (status === 'running') statusPill.classList.add('running');
	else if (status === 'idle') statusPill.classList.add('idle');
	else if (status === 'completed') statusPill.classList.add('done');
	else if (status === 'error') statusPill.classList.add('error');
	if (status === 'running' && errorBox) errorBox.textContent = 'Processing… this may take a while depending on video length.';
}

// Guarded form listener to avoid runtime errors if element missing
if (form) form.addEventListener('submit', async (e) => {
	e.preventDefault();
	if (errorBox) errorBox.textContent = '';
	const fd = new FormData();
	fd.append('user', document.getElementById('user').value || 'User');
	const file = document.getElementById('video').files[0];
	if (!file) { if (errorBox) errorBox.textContent = 'Please choose a video file.'; return; }
	fd.append('file', file);
	startBtn.disabled = true;
	setPill('running');
	let res;
	try {
		res = await fetch('/api/analyze', { method: 'POST', body: fd });
	} catch (err) {
		setPill('error');
		startBtn.disabled = false;
		if (errorBox) errorBox.textContent = 'Network error while starting analysis.';
		return;
	}
	if (!res.ok) {
		setPill('error');
		startBtn.disabled = false;
		try {
			const j = await res.json();
			if (errorBox) errorBox.textContent = j.detail || 'Unable to start analysis.';
		} catch {
			if (errorBox) errorBox.textContent = 'Unable to start analysis.';
		}
		return;
	}
	const data = await res.json();
	try { console.log('analyze ok, session:', data.session_id); } catch {}
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
			if (startBtn) startBtn.disabled = false;
			return;
		}
		if (!res.ok) return;
		const data = await res.json();
		if (!__loggedFirstStatus) { try { console.log('status:', data); } catch {} __loggedFirstStatus = true; }
		renderStatus(data);
		if (data.status === 'completed' || data.status === 'error') {
			clearInterval(pollTimer);
			if (startBtn) startBtn.disabled = false;
		}
		if (data.status === 'running') {
			refreshScreenshots('');
		}
	}, 1000);
}

async function renderStatus(data) {
	setPill(data.status);
	if (data.status === 'error' && errorBox) {
		errorBox.textContent = data.error || 'Analysis failed.';
	}
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