from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any


_STYLE = """
:root {
  color-scheme: dark;
  --bg: #071018;
  --panel: rgba(15, 28, 40, .88);
  --panel2: #102334;
  --line: #29455a;
  --text: #e8f1f7;
  --muted: #91a8b8;
  --cyan: #42d9d0;
  --yellow: #f5c451;
  --red: #ff7185;
  --green: #75e6a4;
  --blue: #76a9ff;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background:
    radial-gradient(circle at 18% 0%, rgba(47, 120, 150, .22), transparent 38rem),
    radial-gradient(circle at 100% 10%, rgba(84, 54, 140, .18), transparent 34rem), var(--bg);
  color: var(--text);
}
main { width: min(1500px, calc(100% - 32px)); margin: 0 auto 70px; }
header { padding: 34px 0 22px; display: flex; justify-content: space-between; gap: 24px; align-items: end; }
.eyebrow { color: var(--cyan); text-transform: uppercase; letter-spacing: .16em; font-size: 12px; font-weight: 800; }
h1 { margin: 8px 0 5px; font-size: clamp(27px, 4vw, 46px); line-height: 1.05; }
h2 { margin: 0 0 16px; font-size: 19px; }
h3 { margin: 0 0 8px; font-size: 15px; }
p { color: var(--muted); line-height: 1.65; }
a { color: var(--cyan); text-decoration: none; }
a:hover { text-decoration: underline; }
.nav { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }
.nav a { border: 1px solid var(--line); background: rgba(8, 20, 30, .7); padding: 8px 12px; border-radius: 999px; }
.metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 10px 0 18px; }
.metric, .panel { border: 1px solid var(--line); background: var(--panel); box-shadow: 0 14px 36px rgba(0,0,0,.22); }
.metric { border-radius: 14px; padding: 16px; min-height: 98px; }
.metric .label { font-size: 12px; color: var(--muted); }
.metric .value { font-size: 24px; font-weight: 800; margin-top: 10px; overflow-wrap: anywhere; }
.metric .detail { margin-top: 4px; font-size: 12px; color: var(--muted); }
.panel { border-radius: 16px; padding: 18px; margin: 14px 0; overflow: hidden; }
.grid2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
.grid3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
.episode-block { border: 1px solid var(--line); background: var(--panel); border-radius: 18px; padding: 20px; margin: 16px 0; box-shadow: 0 14px 36px rgba(0,0,0,.22); }
.episode-head { display: flex; align-items: center; justify-content: space-between; gap: 14px; margin-bottom: 16px; }
.episode-head h2 { margin: 0; font-size: 24px; }
.episode-qa { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }
.stage-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
.stage-card { display: flex; min-width: 0; flex-direction: column; overflow: hidden; border: 1px solid var(--line); border-radius: 15px; background: #07131d; color: var(--text); text-decoration: none; transition: transform .15s ease, border-color .15s ease, box-shadow .15s ease; }
.stage-card:hover { text-decoration: none; transform: translateY(-3px); border-color: var(--cyan); box-shadow: 0 14px 30px rgba(0,0,0,.34); }
.stage-preview { position: relative; display: grid; place-items: center; aspect-ratio: 16/9; overflow: hidden; background: linear-gradient(135deg, #102b3c, #111728); border-bottom: 1px solid var(--line); }
.stage-preview img { width: 100%; height: 100%; display: block; object-fit: cover; }
.stage-placeholder { font-size: 34px; font-weight: 900; letter-spacing: .08em; color: rgba(232,241,247,.38); }
.stage-number { position: absolute; left: 10px; top: 10px; padding: 5px 8px; border: 1px solid rgba(255,255,255,.22); border-radius: 7px; background: rgba(0,0,0,.68); color: #fff; font: 800 11px ui-monospace, monospace; }
.stage-content { display: flex; flex: 1; flex-direction: column; gap: 9px; padding: 14px; }
.stage-title { display: flex; justify-content: space-between; align-items: center; gap: 9px; }
.stage-title strong { font-size: 17px; }
.stage-content p { margin: 0; font-size: 12px; line-height: 1.55; }
.stage-open { margin-top: auto; color: var(--cyan); font-size: 13px; font-weight: 800; }
.status { display: inline-flex; align-items: center; gap: 7px; border: 1px solid; padding: 5px 9px; border-radius: 999px; font-size: 12px; font-weight: 750; }
.status::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
.ok { color: var(--green); background: rgba(40,140,90,.12); }
.bad { color: var(--red); background: rgba(170,50,70,.12); }
.warn { color: var(--yellow); background: rgba(190,135,30,.12); }
.muted { color: var(--muted); }
.chart-wrap { position: relative; min-height: 310px; }
canvas { width: 100%; display: block; border-radius: 10px; background: rgba(5, 15, 23, .65); }
canvas.line-chart { height: 300px; }
canvas.scene3d { height: 470px; cursor: grab; }
canvas.scene3d:active { cursor: grabbing; }
canvas.heatmap-canvas { height: 350px; }
.data-player { display: grid; grid-template-columns: minmax(0, 1.7fr) minmax(270px, .7fr); gap: 14px; }
.frame-stage { position: relative; min-height: 360px; display: grid; place-items: center; overflow: hidden; border: 1px solid var(--line); border-radius: 12px; background: #020609; }
.frame-stage img { display: block; width: 100%; max-height: 72vh; object-fit: contain; }
.frame-stage canvas { position: absolute; inset: 0; width: 100%; height: 100%; background: transparent; border-radius: 0; pointer-events: none; }
.player-side { display: flex; flex-direction: column; gap: 10px; }
.player-toolbar { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; }
.player-toolbar button, .player-toolbar select {
  color: var(--text); background: var(--panel2); border: 1px solid var(--line); border-radius: 8px; padding: 7px 10px;
}
.player-toolbar button { cursor: pointer; min-width: 38px; }
.player-toolbar button:hover { border-color: var(--cyan); }
.player-timeline { display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: center; }
.player-timeline input { width: 100%; accent-color: var(--cyan); }
.layer-controls { display: flex; flex-wrap: wrap; gap: 7px; }
.layer-toggle { display: inline-flex; gap: 6px; align-items: center; border: 1px solid var(--line); border-radius: 999px; padding: 5px 9px; color: #c7d7e1; font-size: 12px; }
.layer-toggle input { accent-color: var(--cyan); }
.telemetry { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 7px; }
.telemetry-item { border: 1px solid rgba(64,94,114,.55); border-radius: 9px; padding: 8px 9px; background: rgba(5,15,23,.55); min-width: 0; }
.telemetry-item .k { display: block; color: var(--muted); font-size: 10px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.telemetry-item .v { display: block; margin-top: 3px; color: #f1f7fa; font: 700 13px ui-monospace, monospace; overflow-wrap: anywhere; }
canvas.skeleton3d { height: 520px; cursor: grab; }
canvas.skeleton3d:active { cursor: grabbing; }
canvas.vector-canvas { height: 360px; }
.vector-values { display: grid; grid-template-columns: repeat(auto-fit, minmax(112px, 1fr)); gap: 6px; margin-top: 10px; }
.vector-value { border-left: 3px solid var(--cyan); background: rgba(5,15,23,.55); padding: 6px 8px; border-radius: 5px; }
.vector-value .name { color: var(--muted); font-size: 10px; }
.vector-value .number { font: 700 12px ui-monospace, monospace; margin-top: 2px; }
.control-row { display: flex; align-items: center; gap: 12px; margin: 12px 0 4px; }
.control-row input[type=range] { flex: 1; accent-color: var(--cyan); }
.frame-label { min-width: 110px; text-align: right; color: var(--muted); font-variant-numeric: tabular-nums; }
.gallery { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px; }
.frame { background: #050b10; border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
.frame img { display: block; width: 100%; aspect-ratio: 4/3; object-fit: contain; }
.frame figcaption { color: var(--muted); font-size: 12px; padding: 8px 10px; }
.compare { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.compare figure { margin: 0; border: 1px solid var(--line); border-radius: 12px; overflow: hidden; background: #050b10; }
.compare img { display: block; width: 100%; aspect-ratio: 4/3; object-fit: contain; }
.compare figcaption { padding: 8px 10px; color: var(--muted); }
video { width: 100%; max-height: 650px; background: #000; border-radius: 12px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { text-align: left; padding: 10px; border-bottom: 1px solid rgba(64,94,114,.45); vertical-align: top; }
th { color: #bad0de; font-weight: 700; position: sticky; top: 0; background: var(--panel2); }
td { color: #dce8ef; }
code { color: #bcebe8; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .92em; }
.scroll { overflow: auto; max-height: 560px; }
.notice { border-left: 3px solid var(--yellow); padding: 10px 14px; background: rgba(160,110,20,.1); color: #f3d998; }
.legend-note { font-size: 12px; color: var(--muted); margin-top: 8px; }
.generated-at { display: inline-block; margin: 8px 0 0; padding: 5px 9px; border: 1px solid var(--line); border-radius: 7px; color: var(--yellow); font: 700 11px ui-monospace, monospace; }
footer { color: var(--muted); font-size: 12px; border-top: 1px solid var(--line); padding-top: 18px; margin-top: 30px; }
@media (max-width: 820px) {
  .grid2, .grid3, .compare, .data-player, .stage-grid { grid-template-columns: 1fr; }
  .episode-head { align-items: flex-start; flex-direction: column; }
  .episode-qa { justify-content: flex-start; }
  header { display: block; }
  .nav { justify-content: flex-start; margin-top: 16px; }
  main { width: min(100% - 18px, 1500px); }
}
"""


_SCRIPT = r"""
const DATA = window.REPORT_DATA || {};
const COLORS = ['#42d9d0','#f5c451','#ff7185','#76a9ff','#75e6a4','#d697ff','#ff9f68','#c8e66b','#5ed0ff','#e98bac','#b7c5d0','#ffffff'];
const FRAME_STATE = {};

function publishFrame(group, frame, time, source) {
  if (!group) return;
  FRAME_STATE[group] = {frame, time};
  window.dispatchEvent(new CustomEvent('report-frame', {detail:{group, frame, time, source}}));
}

function setupCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(320, rect.width);
  const height = Math.max(220, parseFloat(getComputedStyle(canvas).height));
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return {ctx, width, height};
}

function finite(v) { return Number.isFinite(v); }
function nearestTimeIndex(times, target, fallback=0) {
  if (!Array.isArray(times) || !times.length || !finite(target)) return fallback;
  let lo=0, hi=times.length-1;
  while(lo<hi){const mid=Math.floor((lo+hi)/2);if(times[mid]<target)lo=mid+1;else hi=mid;}
  if(lo>0 && Math.abs(times[lo-1]-target)<=Math.abs(times[lo]-target))return lo-1;
  return lo;
}
function fmt(v) {
  if (!finite(v)) return '—';
  const a = Math.abs(v);
  if (a >= 1000 || (a > 0 && a < .001)) return v.toExponential(2);
  return v.toFixed(a < 1 ? 3 : 2);
}

function drawLineChart(canvas) {
  const spec = (DATA.charts || {})[canvas.dataset.lineChart];
  if (!spec) return;
  const {ctx, width:w, height:h} = setupCanvas(canvas);
  const m = {l:58,r:18,t:42,b:42};
  const allY = spec.series.flatMap(s => s.y).filter(finite);
  if (finite(spec.threshold)) allY.push(spec.threshold);
  let ymin = finite(spec.ymin) ? spec.ymin : Math.min(...allY);
  let ymax = finite(spec.ymax) ? spec.ymax : Math.max(...allY);
  if (!finite(ymin) || !finite(ymax)) { ymin=0; ymax=1; }
  if (Math.abs(ymax-ymin) < 1e-9) { ymin -= .5; ymax += .5; }
  const pad=(ymax-ymin)*.08; ymin-=pad; ymax+=pad;
  const x = spec.x || spec.series[0].y.map((_,i)=>i);
  const xmin=Math.min(...x), xmax=Math.max(...x);
  const px = v => m.l + (v-xmin)/Math.max(xmax-xmin,1e-9)*(w-m.l-m.r);
  const py = v => h-m.b - (v-ymin)/(ymax-ymin)*(h-m.t-m.b);
  ctx.clearRect(0,0,w,h);
  ctx.font='12px system-ui'; ctx.lineWidth=1;
  for(let i=0;i<=5;i++) {
    const yy=m.t+i*(h-m.t-m.b)/5;
    ctx.strokeStyle='rgba(90,125,145,.22)'; ctx.beginPath(); ctx.moveTo(m.l,yy); ctx.lineTo(w-m.r,yy); ctx.stroke();
    const val=ymax-i*(ymax-ymin)/5; ctx.fillStyle='#91a8b8'; ctx.textAlign='right'; ctx.fillText(fmt(val),m.l-8,yy+4);
  }
  for(let i=0;i<=5;i++) {
    const xx=m.l+i*(w-m.l-m.r)/5; const val=xmin+i*(xmax-xmin)/5;
    ctx.fillStyle='#91a8b8'; ctx.textAlign='center'; ctx.fillText(fmt(val),xx,h-17);
  }
  if (finite(spec.threshold)) {
    const yy=py(spec.threshold); ctx.setLineDash([7,5]); ctx.strokeStyle='#ff7185'; ctx.beginPath(); ctx.moveTo(m.l,yy); ctx.lineTo(w-m.r,yy); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle='#ff7185'; ctx.textAlign='left'; ctx.fillText(`阈值 ${fmt(spec.threshold)}`,m.l+5,yy-6);
  }
  spec.series.forEach((s,si)=>{
    ctx.strokeStyle=s.color || COLORS[si%COLORS.length]; ctx.lineWidth=1.7; ctx.beginPath();
    let started=false;
    s.y.forEach((v,i)=>{ if(!finite(v)) {started=false; return;} const xx=px(x[i]), yy=py(v); if(!started){ctx.moveTo(xx,yy);started=true;}else ctx.lineTo(xx,yy); });
    ctx.stroke();
  });
  const cursor=spec.syncGroup && FRAME_STATE[spec.syncGroup];
  if(cursor && finite(cursor.time)) {
    const xx=px(Math.max(xmin,Math.min(xmax,cursor.time)));
    ctx.strokeStyle='#ffffff';ctx.globalAlpha=.72;ctx.lineWidth=1;ctx.setLineDash([3,4]);ctx.beginPath();ctx.moveTo(xx,m.t);ctx.lineTo(xx,h-m.b);ctx.stroke();ctx.setLineDash([]);ctx.globalAlpha=1;
    ctx.fillStyle='#ffffff';ctx.textAlign='center';ctx.font='10px ui-monospace';ctx.fillText(`t=${fmt(cursor.time)}s`,xx,m.t-7);
  }
  ctx.fillStyle='#e8f1f7'; ctx.textAlign='left'; ctx.font='700 14px system-ui'; ctx.fillText(spec.title || '',m.l,20);
  ctx.font='11px system-ui'; let lx=m.l;
  spec.series.forEach((s,si)=>{ const color=s.color||COLORS[si%COLORS.length]; ctx.fillStyle=color; ctx.fillRect(lx,28,12,3); ctx.fillStyle='#b9ccd8'; ctx.fillText(s.name,lx+17,33); lx+=22+ctx.measureText(s.name).width; });
  ctx.fillStyle='#91a8b8'; ctx.textAlign='right'; ctx.fillText(spec.xLabel || 'time (s)',w-m.r,h-3);
  if(spec.yLabel){ctx.save();ctx.translate(12,m.t);ctx.rotate(-Math.PI/2);ctx.textAlign='right';ctx.fillText(spec.yLabel,0,0);ctx.restore();}
}

function initTimelinePlayer(wrapper) {
  const spec=(DATA.timelinePlayers||{})[wrapper.dataset.timelinePlayer];
  if(!spec || !spec.frames || !spec.frames.length) return;
  const img=wrapper.querySelector('.frame-stage img'), overlay=wrapper.querySelector('.frame-stage canvas');
  const slider=wrapper.querySelector('.player-timeline input'), label=wrapper.querySelector('.frame-label');
  const play=wrapper.querySelector('[data-action="play"]'), prev=wrapper.querySelector('[data-action="prev"]'), next=wrapper.querySelector('[data-action="next"]');
  const speed=wrapper.querySelector('select'), telemetry=wrapper.querySelector('.telemetry'), layerBox=wrapper.querySelector('.layer-controls');
  const layers=spec.layers||[]; let frame=0, timer=null, playing=false;
  slider.max=spec.frames.length-1;
  layers.forEach((layer,index)=>{
    const item=document.createElement('label');item.className='layer-toggle';item.style.borderColor=layer.color||COLORS[index%COLORS.length];
    item.innerHTML=`<input type="checkbox" checked data-layer="${index}"><span>${layer.name||`layer ${index}`}</span>`;layerBox.appendChild(item);
  });
  function frameTime(i){return spec.times&&finite(spec.times[i])?spec.times[i]:i/Math.max(spec.fps||30,1);}
  function renderTelemetry(i){telemetry.innerHTML='';(spec.telemetry||[]).forEach(row=>{const raw=Array.isArray(row.values)?row.values[Math.min(i,row.values.length-1)]:row.value;const value=typeof raw==='number'?fmt(raw):String(raw??'—');const node=document.createElement('div');node.className='telemetry-item';node.innerHTML=`<span class="k">${row.label}</span><span class="v">${value}${row.unit?` ${row.unit}`:''}</span>`;telemetry.appendChild(node);});}
  function drawOverlay(){
    const rect=overlay.getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio||1,2),w=Math.max(rect.width,1),h=Math.max(rect.height,1);overlay.width=Math.round(w*dpr);overlay.height=Math.round(h*dpr);const ctx=overlay.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
    const sourceW=spec.width||img.naturalWidth||w,sourceH=spec.height||img.naturalHeight||h,scale=Math.min(w/sourceW,h/sourceH),ox=(w-sourceW*scale)/2,oy=(h-sourceH*scale)/2;
    layers.forEach((layer,li)=>{const toggle=layerBox.querySelector(`[data-layer="${li}"]`);if(toggle&&!toggle.checked)return;if(layer.valid&&layer.valid[frame]===false)return;const points=layer.points&&layer.points[frame];if(!points)return;const color=layer.color||COLORS[li%COLORS.length];ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineCap='round';ctx.lineJoin='round';ctx.lineWidth=layer.lineWidth||2.2;
      (layer.bones||[]).forEach(pair=>{const a=points[pair[0]],b=points[pair[1]];if(!a||!b||!finite(a[0])||!finite(a[1])||!finite(b[0])||!finite(b[1]))return;ctx.beginPath();ctx.moveTo(ox+a[0]*scale,oy+a[1]*scale);ctx.lineTo(ox+b[0]*scale,oy+b[1]*scale);ctx.stroke();});
      points.forEach((p,pi)=>{if(!p||!finite(p[0])||!finite(p[1]))return;ctx.beginPath();ctx.arc(ox+p[0]*scale,oy+p[1]*scale,pi===0?4.2:2.5,0,Math.PI*2);ctx.fill();});
    });
    ctx.fillStyle='rgba(0,0,0,.58)';ctx.fillRect(10,10,184,27);ctx.fillStyle='#fff';ctx.font='12px ui-monospace';ctx.fillText(`${spec.frameName||'frame'} ${frame}  t=${fmt(frameTime(frame))}s`,18,28);
  }
  function setFrame(value, announce=true){frame=Math.max(0,Math.min(spec.frames.length-1,Number(value)||0));slider.value=frame;label.textContent=`frame ${frame} / ${spec.frames.length-1}`;img.src=spec.frames[frame];renderTelemetry(frame);drawOverlay();if(announce)publishFrame(spec.syncGroup,frame,frameTime(frame),wrapper);const preload=new Image();preload.src=spec.frames[Math.min(frame+1,spec.frames.length-1)];}
  function stop(){playing=false;play.textContent='▶';if(timer){clearTimeout(timer);timer=null;}}
  function tick(){if(!playing)return;if(frame>=spec.frames.length-1){stop();return;}setFrame(frame+1);timer=setTimeout(tick,1000/Math.max((spec.fps||30)*Number(speed.value||1),1));}
  play.addEventListener('click',()=>{if(playing){stop();return;}playing=true;play.textContent='⏸';tick();});prev.addEventListener('click',()=>{stop();setFrame(frame-1);});next.addEventListener('click',()=>{stop();setFrame(frame+1);});slider.addEventListener('input',()=>{stop();setFrame(slider.value);});layerBox.addEventListener('change',drawOverlay);img.addEventListener('load',drawOverlay);new ResizeObserver(drawOverlay).observe(wrapper.querySelector('.frame-stage'));
  wrapper.addEventListener('keydown',e=>{if(e.key==='ArrowLeft'){e.preventDefault();stop();setFrame(frame-1);}else if(e.key==='ArrowRight'){e.preventDefault();stop();setFrame(frame+1);}else if(e.key===' '){e.preventDefault();play.click();}});
  window.addEventListener('report-frame',e=>{if(e.detail.group===spec.syncGroup&&e.detail.source!==wrapper)setFrame(nearestTimeIndex(spec.times,e.detail.time,e.detail.frame),false);});setFrame(0);
}

function initSkeleton3d(wrapper){
  const spec=(DATA.skeletons3d||{})[wrapper.dataset.skeleton3d];if(!spec)return;const canvas=wrapper.querySelector('canvas'),slider=wrapper.querySelector('input'),label=wrapper.querySelector('.frame-label');const all=spec.skeletons||[];const count=spec.frames||Math.max(0,...all.map(s=>s.points.length));slider.max=Math.max(0,count-1);let frame=0,yaw=-.65,pitch=.25;
  const flat=[];all.forEach(s=>s.points.forEach(ps=>(ps||[]).forEach(p=>{if(p&&p.every(finite))flat.push(p)})));if(!flat.length)return;const mins=[0,1,2].map(k=>Math.min(...flat.map(p=>p[k]))),maxs=[0,1,2].map(k=>Math.max(...flat.map(p=>p[k])));const center=mins.map((v,k)=>(v+maxs[k])/2),span=Math.max(...maxs.map((v,k)=>v-mins[k]),.15);
  function project(p,w,h){const q=rotPoint(p.map((v,k)=>v-center[k]),yaw,pitch);return [w/2+q[0]/span*w*.72,h/2-q[1]/span*w*.72,q[2]];}
  function draw(){const {ctx,width:w,height:h}=setupCanvas(canvas);ctx.clearRect(0,0,w,h);label.textContent=`frame ${frame} / ${count-1}`;ctx.strokeStyle='rgba(100,135,155,.17)';ctx.lineWidth=1;for(let i=-5;i<=5;i++){const a=project([center[0]+i*span/10,mins[1],center[2]-span/2],w,h),b=project([center[0]+i*span/10,mins[1],center[2]+span/2],w,h);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();}
    all.forEach((s,si)=>{const pts=s.points[Math.min(frame,s.points.length-1)]||[],color=s.color||COLORS[si%COLORS.length];if(s.valid&&s.valid[Math.min(frame,s.valid.length-1)]===false)return;const root=Number.isInteger(s.trailJoint)?s.trailJoint:0;ctx.strokeStyle=color;ctx.globalAlpha=.3;ctx.lineWidth=1;ctx.beginPath();let begun=false;for(let f=0;f<=frame;f++){const p=s.points[f]&&s.points[f][root];if(!p||!p.every(finite))continue;const q=project(p,w,h);if(!begun){ctx.moveTo(q[0],q[1]);begun=true;}else ctx.lineTo(q[0],q[1]);}ctx.stroke();ctx.globalAlpha=1;ctx.strokeStyle=color;ctx.lineWidth=2.5;(s.bones||[]).forEach(b=>{const a=pts[b[0]],c=pts[b[1]];if(!a||!c||!a.every(finite)||!c.every(finite))return;const pa=project(a,w,h),pc=project(c,w,h);ctx.beginPath();ctx.moveTo(pa[0],pa[1]);ctx.lineTo(pc[0],pc[1]);ctx.stroke();});pts.forEach((p,i)=>{if(!p||!p.every(finite))return;const q=project(p,w,h);ctx.fillStyle=color;ctx.beginPath();ctx.arc(q[0],q[1],i===root?5:3,0,Math.PI*2);ctx.fill();});const anchor=pts[root];if(anchor&&anchor.every(finite)){const q=project(anchor,w,h);ctx.fillStyle='#e8f1f7';ctx.font='12px system-ui';ctx.fillText(s.name||`skeleton ${si}`,q[0]+7,q[1]-7);}});ctx.fillStyle='#91a8b8';ctx.font='11px system-ui';ctx.fillText(spec.note||'拖拽旋转；滚动条与图像播放器同步',14,h-14);}
  function setFrame(i,announce=true){frame=Math.max(0,Math.min(count-1,Number(i)||0));slider.value=frame;draw();if(announce){const t=spec.times&&spec.times[frame];publishFrame(spec.syncGroup,frame,finite(t)?t:null,wrapper);}}slider.addEventListener('input',()=>setFrame(slider.value));let down=false,last=[0,0];canvas.addEventListener('pointerdown',e=>{down=true;last=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);});canvas.addEventListener('pointerup',()=>down=false);canvas.addEventListener('pointermove',e=>{if(!down)return;yaw+=(e.clientX-last[0])*.008;pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-last[1])*.008));last=[e.clientX,e.clientY];draw();});window.addEventListener('report-frame',e=>{if(e.detail.group===spec.syncGroup&&e.detail.source!==wrapper)setFrame(nearestTimeIndex(spec.times,e.detail.time,e.detail.frame),false);});new ResizeObserver(draw).observe(canvas);setFrame(0,false);
}

function initVectorViewer(wrapper){
  const spec=(DATA.vectorViews||{})[wrapper.dataset.vectorView];if(!spec||!spec.current||!spec.current.length)return;const canvas=wrapper.querySelector('canvas'),slider=wrapper.querySelector('input'),label=wrapper.querySelector('.frame-label'),values=wrapper.querySelector('.vector-values');const count=spec.current.length;slider.max=count-1;let frame=0;const groups=spec.groups||[{name:'vector',start:0,end:spec.labels.length,color:COLORS[0]}];const scales=groups.map(g=>{let m=.001;for(let f=0;f<count;f++)for(let i=g.start;i<g.end;i++){m=Math.max(m,Math.abs(spec.current[f][i]||0),Math.abs((spec.target&&spec.target[f]&&spec.target[f][i])||0));}return m;});
  function draw(){const {ctx,width:w,height:h}=setupCanvas(canvas);ctx.clearRect(0,0,w,h);const n=spec.labels.length,left=22,right=14,top=34,bottom=52,cw=(w-left-right)/n,mid=top+(h-top-bottom)/2;ctx.strokeStyle='rgba(120,150,170,.35)';ctx.beginPath();ctx.moveTo(left,mid);ctx.lineTo(w-right,mid);ctx.stroke();ctx.fillStyle='#e8f1f7';ctx.font='700 13px system-ui';ctx.fillText(spec.title||'current / next action',left,18);values.innerHTML='';groups.forEach((g,gi)=>{const color=g.color||COLORS[gi%COLORS.length],scale=scales[gi],startX=left+g.start*cw,endX=left+g.end*cw;ctx.fillStyle=color;ctx.globalAlpha=.1;ctx.fillRect(startX,top,endX-startX,h-top-bottom);ctx.globalAlpha=1;ctx.fillStyle=color;ctx.textAlign='center';ctx.font='10px system-ui';ctx.fillText(g.name,(startX+endX)/2,h-8);for(let i=g.start;i<g.end;i++){const v=spec.current[frame][i]||0,t=spec.target&&spec.target[frame]?spec.target[frame][i]:null,x=left+(i+.5)*cw,bar=v/scale*(h-top-bottom)*.43;ctx.fillStyle=color;ctx.globalAlpha=.78;ctx.fillRect(x-cw*.28,Math.min(mid,mid-bar),cw*.34,Math.abs(bar));ctx.globalAlpha=1;if(finite(t)){const ty=mid-t/scale*(h-top-bottom)*.43;ctx.strokeStyle='#fff';ctx.lineWidth=1.3;ctx.strokeRect(x+cw*.02,ty-2,cw*.34,4);}const node=document.createElement('div');node.className='vector-value';node.style.borderColor=color;node.innerHTML=`<div class="name">${spec.labels[i]}</div><div class="number">${fmt(v)}${finite(t)?` → ${fmt(t)}`:''}</div>`;values.appendChild(node);}});ctx.fillStyle='#b9ccd8';ctx.textAlign='right';ctx.font='10px system-ui';ctx.fillText('实心=current，白框=action[t]=state[t+1]',w-right,18);label.textContent=`frame ${frame} / ${count-1}`;}
  function setFrame(i,announce=true){frame=Math.max(0,Math.min(count-1,Number(i)||0));slider.value=frame;draw();if(announce){const t=spec.times&&spec.times[frame];publishFrame(spec.syncGroup,frame,finite(t)?t:null,wrapper);}}slider.addEventListener('input',()=>setFrame(slider.value));window.addEventListener('report-frame',e=>{if(e.detail.group===spec.syncGroup&&e.detail.source!==wrapper)setFrame(nearestTimeIndex(spec.times,e.detail.time,e.detail.frame),false);});new ResizeObserver(draw).observe(canvas);setFrame(0,false);
}

function rotPoint(p,yaw,pitch) {
  const cy=Math.cos(yaw), sy=Math.sin(yaw), cp=Math.cos(pitch), sp=Math.sin(pitch);
  const x=cy*p[0]-sy*p[2], z=sy*p[0]+cy*p[2];
  return [x, cp*p[1]-sp*z, sp*p[1]+cp*z];
}
function initScene(wrapper) {
  const spec=(DATA.scenes||{})[wrapper.dataset.scene]; if(!spec) return;
  const canvas=wrapper.querySelector('canvas'), slider=wrapper.querySelector('input'), label=wrapper.querySelector('.frame-label');
  slider.max=Math.max(0,spec.frames-1); let yaw=-.65,pitch=.42;
  const points=spec.entities.flatMap(e=>e.positions).filter(p=>p&&p.every(finite));
  (spec.referencePoints||[]).forEach(p=>{if(p&&p.every(finite))points.push(p);});
  const mins=[0,1,2].map(k=>Math.min(...points.map(p=>p[k]))), maxs=[0,1,2].map(k=>Math.max(...points.map(p=>p[k])));
  const center=mins.map((v,k)=>(v+maxs[k])/2), span=Math.max(...maxs.map((v,k)=>v-mins[k]),.1);
  function project(p,w,h){const q=rotPoint(p.map((v,k)=>v-center[k]),yaw,pitch);return [w/2+q[0]/span*w*.72,h/2-q[1]/span*w*.72,q[2]];}
  function drawArrow(ctx,a,b,color,label){
    ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=2.4;ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();
    const ang=Math.atan2(b[1]-a[1],b[0]-a[0]),len=9;
    ctx.beginPath();ctx.moveTo(b[0],b[1]);ctx.lineTo(b[0]-len*Math.cos(ang-.45),b[1]-len*Math.sin(ang-.45));ctx.lineTo(b[0]-len*Math.cos(ang+.45),b[1]-len*Math.sin(ang+.45));ctx.closePath();ctx.fill();
    if(label){ctx.font='700 12px system-ui';ctx.fillText(label,b[0]+5,b[1]-5);}
  }
  function draw(){
    const {ctx,width:w,height:h}=setupCanvas(canvas);ctx.clearRect(0,0,w,h);const frame=Number(slider.value);label.textContent=`frame ${frame} / ${spec.frames-1}`;
    ctx.strokeStyle='rgba(100,135,155,.18)';ctx.lineWidth=1;
    for(let i=-5;i<=5;i++){const a=project([center[0]+i*span/10,center[1],center[2]-span/2],w,h),b=project([center[0]+i*span/10,center[1],center[2]+span/2],w,h);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();}
    if(spec.baseFrame){
      const o=spec.baseFrame.origin||[0,0,0],len=spec.baseFrame.axisLength||span*.22,po=project(o,w,h);
      drawArrow(ctx,po,project([o[0]+len,o[1],o[2]],w,h),'#ff7185','x');
      drawArrow(ctx,po,project([o[0],o[1]+len,o[2]],w,h),'#75e6a4','y');
      drawArrow(ctx,po,project([o[0],o[1],o[2]+len],w,h),'#76a9ff','z');
      ctx.fillStyle='#e8f1f7';ctx.font='700 12px system-ui';ctx.fillText(spec.baseFrame.name||'robot base',po[0]+8,po[1]+16);
    }
    spec.entities.forEach((e,ei)=>{
      const color=e.color||COLORS[ei%COLORS.length];ctx.strokeStyle=color;ctx.globalAlpha=e.trailAlpha??.38;ctx.lineWidth=e.lineWidth||1.4;ctx.setLineDash(e.dash||[]);
      if(e.trail!==false){ctx.beginPath();let start=false;e.positions.slice(0,frame+1).forEach(p=>{if(!p||!p.every(finite))return;const q=project(p,w,h);if(!start){ctx.moveTo(q[0],q[1]);start=true;}else ctx.lineTo(q[0],q[1]);});ctx.stroke();}
      ctx.setLineDash([]);ctx.globalAlpha=1;
      const p=e.positions[Math.min(frame,e.positions.length-1)];if(!p||!p.every(finite))return;const q=project(p,w,h);ctx.globalAlpha=e.alpha??1;ctx.fillStyle=color;ctx.beginPath();ctx.arc(q[0],q[1],e.radius||(e.kind==='hand'?7:6),0,Math.PI*2);ctx.fill();ctx.globalAlpha=1;ctx.fillStyle='#e8f1f7';ctx.font='12px system-ui';ctx.fillText(e.name,q[0]+9,q[1]-8);
      const axes=e.axes&&e.axes[Math.min(frame,e.axes.length-1)];if(axes){['#ff7185','#75e6a4','#76a9ff'].forEach((c,k)=>{const axisLen=e.axisLength||span*.08,end=p.map((v,j)=>v+axes[k][j]*axisLen),qe=project(end,w,h);ctx.strokeStyle=c;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(q[0],q[1]);ctx.lineTo(qe[0],qe[1]);ctx.stroke();});}
    });
    ctx.fillStyle='#91a8b8';ctx.font='11px system-ui';ctx.fillText(spec.note||'拖拽旋转视角；轨迹单位 m；RGB 线为物体/手坐标轴',14,h-14);
  }
  function setFrame(i,announce=true){slider.value=Math.max(0,Math.min(spec.frames-1,Number(i)||0));draw();if(announce){const f=Number(slider.value),t=spec.times&&spec.times[f];publishFrame(spec.syncGroup,f,finite(t)?t:null,wrapper);}}
  slider.addEventListener('input',()=>setFrame(slider.value)); let down=false,last=[0,0]; canvas.addEventListener('pointerdown',e=>{down=true;last=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);});canvas.addEventListener('pointerup',()=>down=false);canvas.addEventListener('pointermove',e=>{if(!down)return;yaw+=(e.clientX-last[0])*.008;pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-last[1])*.008));last=[e.clientX,e.clientY];draw();});window.addEventListener('report-frame',e=>{if(e.detail.group===spec.syncGroup&&e.detail.source!==wrapper)setFrame(nearestTimeIndex(spec.times,e.detail.time,e.detail.frame),false);});
  new ResizeObserver(draw).observe(canvas);setFrame(0,false);
}

function initHeatmap(wrapper){
  const spec=(DATA.heatmaps||{})[wrapper.dataset.heatmap];if(!spec)return;const canvas=wrapper.querySelector('canvas'),slider=wrapper.querySelector('input'),label=wrapper.querySelector('.frame-label');slider.max=Math.max(0,spec.frames.length-1);
  function draw(){const {ctx,width:w,height:h}=setupCanvas(canvas),f=Number(slider.value),mat=spec.frames[f]||[],rows=mat.length,cols=rows?mat[0].length:0;label.textContent=`frame ${f} / ${spec.frames.length-1}`;ctx.clearRect(0,0,w,h);if(!rows||!cols)return;const left=74,top=28,cw=(w-left-12)/cols,ch=(h-top-32)/rows;let max=1;mat.forEach(r=>r.forEach(v=>{if(finite(v))max=Math.max(max,Math.abs(v));}));for(let r=0;r<rows;r++)for(let c=0;c<cols;c++){const v=mat[r][c],a=Math.min(1,Math.abs(v)/max),hue=v>=0?174:347;ctx.fillStyle=`hsla(${hue},72%,${25+a*42}%,${.25+a*.75})`;ctx.fillRect(left+c*cw,top+r*ch,Math.ceil(cw),Math.ceil(ch));}ctx.font='11px system-ui';ctx.fillStyle='#a9bdca';ctx.textAlign='right';for(let r=0;r<rows;r++)ctx.fillText((spec.rowLabels||[])[r]||String(r),left-7,top+(r+.68)*ch);ctx.textAlign='center';for(let c=0;c<cols;c+=Math.max(1,Math.ceil(cols/12)))ctx.fillText((spec.colLabels||[])[c]||String(c),left+(c+.5)*cw,h-9);}
  function setFrame(i,announce=true){slider.value=Math.max(0,Math.min(spec.frames.length-1,Number(i)||0));draw();if(announce){const f=Number(slider.value),t=spec.times&&spec.times[f];publishFrame(spec.syncGroup,f,finite(t)?t:null,wrapper);}}slider.addEventListener('input',()=>setFrame(slider.value));window.addEventListener('report-frame',e=>{if(e.detail.group===spec.syncGroup&&e.detail.source!==wrapper)setFrame(nearestTimeIndex(spec.times,e.detail.time,e.detail.frame),false);});new ResizeObserver(draw).observe(canvas);setFrame(0,false);
}

function initCompare(wrapper){const spec=(DATA.comparisons||{})[wrapper.dataset.comparison];if(!spec||!spec.frames.length)return;const slider=wrapper.querySelector('input'),label=wrapper.querySelector('.frame-label'),imgs=wrapper.querySelectorAll('img');slider.max=Math.max(0,spec.frames.length-1);function draw(){const i=Number(slider.value),f=spec.frames[i];imgs[0].src=f.left;imgs[1].src=f.right;label.textContent=f.label||`frame ${i}`;}slider.addEventListener('input',draw);draw();}

function renderAll(){document.querySelectorAll('[data-timeline-player]').forEach(initTimelinePlayer);document.querySelectorAll('[data-skeleton3d]').forEach(initSkeleton3d);document.querySelectorAll('[data-vector-view]').forEach(initVectorViewer);document.querySelectorAll('canvas[data-line-chart]').forEach(c=>{drawLineChart(c);new ResizeObserver(()=>drawLineChart(c)).observe(c)});document.querySelectorAll('[data-scene]').forEach(initScene);document.querySelectorAll('[data-heatmap]').forEach(initHeatmap);document.querySelectorAll('[data-comparison]').forEach(initCompare);window.addEventListener('report-frame',e=>document.querySelectorAll('canvas[data-line-chart]').forEach(c=>{const s=(DATA.charts||{})[c.dataset.lineChart];if(s&&s.syncGroup===e.detail.group)drawLineChart(c);}));}
window.addEventListener('DOMContentLoaded',renderAll);
"""


def _safe_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")


class RawHtml(str):
    """Explicit marker for trusted HTML emitted by report generators."""


def status_badge(text: str, kind: str) -> RawHtml:
    if kind not in {"ok", "bad", "warn"}:
        raise ValueError(kind)
    return RawHtml(f'<span class="status {kind}">{html.escape(text)}</span>')


def metrics(cards: list[tuple[str, str, str]]) -> str:
    return '<div class="metrics">' + "".join(
        '<div class="metric">'
        f'<div class="label">{html.escape(label)}</div>'
        f'<div class="value">{value if isinstance(value, RawHtml) else html.escape(str(value))}</div>'
        f'<div class="detail">{html.escape(detail)}</div>'
        "</div>"
        for label, value, detail in cards
    ) + "</div>"


def panel(title: str, content: str, *, css_class: str = "") -> str:
    class_name = f"panel {css_class}".strip()
    return f'<section class="{class_name}"><h2>{html.escape(title)}</h2>{content}</section>'


def line_chart(key: str) -> str:
    return f'<div class="chart-wrap"><canvas class="line-chart" data-line-chart="{html.escape(key)}"></canvas></div>'


def timeline_player(key: str) -> str:
    return (
        f'<div class="data-player" data-timeline-player="{html.escape(key)}" tabindex="0">'
        '<div class="frame-stage"><img alt="逐帧原始数据"><canvas></canvas></div>'
        '<div class="player-side"><div class="player-toolbar">'
        '<button type="button" data-action="prev" title="上一帧">◀</button>'
        '<button type="button" data-action="play" title="播放/暂停">▶</button>'
        '<button type="button" data-action="next" title="下一帧">▶|</button>'
        '<select title="播放速度"><option value="0.25">0.25×</option><option value="0.5">0.5×</option>'
        '<option value="1" selected>1×</option><option value="2">2×</option></select></div>'
        '<div class="player-timeline"><input type="range" min="0" value="0" step="1">'
        '<span class="frame-label"></span></div><div class="layer-controls"></div>'
        '<div class="telemetry"></div><p class="legend-note">点击播放器后可用空格播放，←/→ 逐帧检查。</p>'
        '</div></div>'
    )


def skeleton3d(key: str) -> str:
    return (
        f'<div data-skeleton3d="{html.escape(key)}"><canvas class="skeleton3d"></canvas>'
        '<div class="control-row"><input type="range" min="0" value="0" step="1">'
        '<span class="frame-label"></span></div></div>'
    )


def vector_view(key: str) -> str:
    return (
        f'<div data-vector-view="{html.escape(key)}"><canvas class="vector-canvas"></canvas>'
        '<div class="control-row"><input type="range" min="0" value="0" step="1">'
        '<span class="frame-label"></span></div><div class="vector-values"></div></div>'
    )


def scene3d(key: str) -> str:
    return (
        f'<div data-scene="{html.escape(key)}">'
        '<canvas class="scene3d"></canvas><div class="control-row">'
        '<input type="range" min="0" value="0" step="1"><span class="frame-label"></span></div></div>'
    )


def heatmap(key: str) -> str:
    return (
        f'<div data-heatmap="{html.escape(key)}">'
        '<canvas class="heatmap-canvas"></canvas><div class="control-row">'
        '<input type="range" min="0" value="0" step="1"><span class="frame-label"></span></div></div>'
    )


def comparison(key: str, left_name: str, right_name: str) -> str:
    return (
        f'<div data-comparison="{html.escape(key)}"><div class="compare">'
        f'<figure><img alt="{html.escape(left_name)}"><figcaption>{html.escape(left_name)}</figcaption></figure>'
        f'<figure><img alt="{html.escape(right_name)}"><figcaption>{html.escape(right_name)}</figcaption></figure>'
        '</div><div class="control-row"><input type="range" min="0" value="0" step="1">'
        '<span class="frame-label"></span></div></div>'
    )


def table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{value if isinstance(value, RawHtml) else html.escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def write_page(
    destination: Path,
    *,
    title: str,
    eyebrow: str,
    subtitle: str,
    body: str,
    data: dict[str, Any] | None = None,
    nav: list[tuple[str, str]] | None = None,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    nav_html = "".join(
        f'<a href="{html.escape(href, quote=True)}">{html.escape(label)}</a>' for label, href in (nav or [])
    )
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Cache-Control" content="no-store, no-cache, must-revalidate"><meta http-equiv="Pragma" content="no-cache">
<title>{html.escape(title)}</title><style>{_STYLE}</style></head>
<body><main><header><div><div class="eyebrow">{html.escape(eyebrow)}</div><h1>{html.escape(title)}</h1>
<p>{html.escape(subtitle)}</p><p class="generated-at">报告生成：{html.escape(generated_at)}</p></div><nav class="nav">{nav_html}</nav></header>{body}
<footer>本页面由 ego_relation_policy 自动生成；所有图表脚本均内嵌，不依赖 CDN。刷新报告请运行 <code>ego-relation report</code>。</footer>
</main><script>window.REPORT_DATA={_safe_json(data or {})};</script><script>{_SCRIPT}</script></body></html>"""
    destination.write_text(page, encoding="utf-8")
    return destination
