(function(){
  const emotionEl = document.getElementById('emotionValue');
  const eyeTurnsEl = document.getElementById('eyeTurnsValue');
  const gazeEl = document.getElementById('gazeValue');
  const yearEl = document.getElementById('year');
  if(yearEl){ yearEl.textContent = new Date().getFullYear(); }

  const barIds = {
    happy: 'barHappy',
    neutral: 'barNeutral',
    sad: 'barSad',
    angry: 'barAngry',
    surprise: 'barSurprise',
    fear: 'barFear',
    disgust: 'barDisgust'
  };

  const bars = Object.fromEntries(Object.entries(barIds).map(([k,id]) => [k, document.getElementById(id)]));

  const emotions = Object.keys(bars);
  let eyeTurns = 0;
  let currentGaze = 'CENTER';
  let gazeHoldStart = Date.now();

  function pickGaze(){
    // Bias toward center; occasionally hold left/right long enough to "count"
    const r = Math.random();
    if(r < 0.68) return 'CENTER';
    if(r < 0.84) return 'LEFT';
    return 'RIGHT';
  }

  function normalize(values){
    const sum = values.reduce((a,b)=>a+b,0) || 1;
    return values.map(v=> (v/sum)*100);
  }

  function updateBars(dist){
    for(const [emo, pct] of Object.entries(dist)){
      const el = bars[emo];
      if(!el) continue;
      el.style.width = `${pct.toFixed(1)}%`;
    }
  }

  function dominantEmotion(dist){
    let best = 'neutral';
    let bestV = -1;
    for(const [k,v] of Object.entries(dist)){
      if(v>bestV){best=k; bestV=v;}
    }
    return best.charAt(0).toUpperCase()+best.slice(1);
  }

  function simulateFrame(){
    // Random emotion distribution with slight continuity
    const base = emotions.map(()=> Math.random() ** 2); // heavier weight to small values
    let distRaw = normalize(base);

    // Slightly boost current dominant to add temporal coherence
    const curDom = (emotionEl?.textContent||'Neutral').toLowerCase();
    if(bars[curDom]){
      const idx = emotions.indexOf(curDom);
      distRaw[idx] = Math.min(100, distRaw[idx] + 10*Math.random());
      // renormalize
      const sum = Object.values(distRaw).reduce((a,b)=>a+b,0);
      distRaw = Object.fromEntries(emotions.map((e,i)=> [e, distRaw[i]/sum*100]));
    } else {
      distRaw = Object.fromEntries(emotions.map((e,i)=> [e, distRaw[i]]));
    }

    // Update bars and primary emotion
    updateBars(distRaw);
    emotionEl.textContent = dominantEmotion(distRaw);

    // Gaze simulation and eye-turn counting (hold >= 2s)
    const newGaze = pickGaze();
    const now = Date.now();
    if(newGaze !== currentGaze){
      currentGaze = newGaze;
      gazeHoldStart = now;
    } else {
      const heldMs = now - gazeHoldStart;
      if((currentGaze === 'LEFT' || currentGaze === 'RIGHT') && heldMs > 2000){
        eyeTurns += 1;
        gazeHoldStart = now + 9999; // prevent rapid recount until direction flips
        eyeTurnsEl.textContent = String(eyeTurns);
      }
    }
    gazeEl.textContent = currentGaze;
  }

  // Initial kick and interval
  simulateFrame();
  setInterval(simulateFrame, 1300);
})();