/* ============================================================
   QUANT·DESK — app.js  (no deps, file:// safe)
   ============================================================ */
const DATA = {
  span: "2016 — 2026 · adjusted daily",
  // real yearly returns from runners/market_park_backtest.py
  years: [2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026],
  spy:   [12.0,21.7,-4.6,31.2,18.3,28.7,-18.2,26.2,24.9,17.7,8.7],
  strat: [10.2,21.6,-1.0,23.7,21.0,27.5,-17.1,26.2,21.5,20.3,6.6],
  stats: [
    {k:"Sharpe",        v:0.97,  d:"vs SPY 0.82",      vc:"var(--up)",  dec:2, sc:"rgba(47,230,166,.12)"},
    {k:"CAGR",          v:13.6,  d:"market + sentinel", vc:"var(--text)",suf:"%",dec:1},
    {k:"Max Drawdown",  v:-22.8, d:"vs SPY −33.7%",     vc:"var(--up)",  suf:"%",dec:1},
    {k:"Walk-forward",  v:5,     d:"of 5 folds positive",vc:"var(--text)",suf:"/5"},
    {k:"Unit tests",    v:63,    d:"rigor · engine · CV",vc:"var(--cyan)",suf:" ✓"},
  ],
  rigor: [
    {l:"Deflated Sharpe (best sleeve)", v:"99.4%", w:"99%", good:1, n:"corrected for 13 trials"},
    {l:"Prob. of Backtest Overfitting", v:"52%",   w:"52%", good:0, n:"single-sleeve pick ≈ noise"},
    {l:"Reality-Check vs SPY (p)",      v:"0.83",  w:"83%", good:0, n:"no alpha over passive"},
    {l:"Overnight edge · net Sharpe",   v:"0.65",  w:"65%", good:1, n:"½ drawdown · DSR 99% · 5/5 WF"},
  ],
  books: [
    {n:"Market + sentinel", note:"idle→market, de-risk on VIX brake", s:0.97, c:13.6, dd:-22.8, tag:"ok"},
    {n:"Equity ensemble", note:"7 sleeves · no-margin · crash sentinel", s:1.59, c:10.5, dd:-7.8, tag:"ok"},
    {n:"Buy & hold SPY", note:"the benchmark we have to beat", s:0.82, c:13.9, dd:-33.7, tag:"bench"},
  ],
  agents: {
    research:[["research_agent","invents equity strategies, web + first-principles"],
              ["autonomous_agent","pure first-principles mechanism search"],
              ["ml_research_agent","proposes daily ML / DL approaches"]],
    build:[["code_agent","spec → signals() module, validation-retry"],
           ["ml_code_agent","ML spec → train-in-signals module"],
           ["options_code_agent","options spec → signals + intent"]],
    validate:[["backtesting_agent","$100k engine + walk-forward + regime"],
              ["risk_agent","gates by Sharpe / DD / win-rate / trades"],
              ["deflated_sharpe","DSR · PBO · reality-check battery"]],
    execute:[["execution_agent","Alpaca paper orders, fractional + vol-target"],
             ["options_agent","Alpaca paper options orders"],
             ["monitor_agent","live P&L + drawdown / concentration alerts"]],
  },
};
const LAYERS = [["research","var(--cyan)","RESEARCH"],["build","var(--violet)","BUILD"],
                ["validate","var(--bench)","VALIDATE"],["execute","var(--up)","EXECUTE"]];
const COL = {research:"#46c8ff",build:"#9d8bff",validate:"#f5a83b",execute:"#2fe6a6"};

const $ = (s,r=document)=>r.querySelector(s);
const $$ = (s,r=document)=>[...r.querySelectorAll(s)];

/* ---------------- routing (with #hash deep-links) ---------------- */
let fieldStop=null;
function activateView(v){
  const b=$(`.nav-item[data-view="${v}"]`);if(!b)v="dashboard";
  $$(".nav-item").forEach(n=>n.classList.toggle("is-active",n.dataset.view===v));
  $$(".view").forEach(x=>x.classList.remove("is-active"));
  $("#view-"+v).classList.add("is-active");
  if(v==="dashboard"){drawEquity();countUp();}
  if(v==="agents"){startField();} else if(fieldStop){fieldStop();fieldStop=null;}
  if(location.hash.slice(1)!==v)history.replaceState(null,"","#"+v);
}
$$(".nav-item").forEach(b=>b.addEventListener("click",()=>activateView(b.dataset.view)));
addEventListener("hashchange",()=>activateView(location.hash.slice(1)||"dashboard"));

/* ---------------- clock + heartbeat ---------------- */
setInterval(()=>{$("#clock").textContent=new Date().toLocaleTimeString("en-US",{hour12:false})+" ET";},1000);
(function heartbeat(){
  const c=$("#pulse"),x=c.getContext("2d");let t=0;
  function rs(){c.width=c.clientWidth*2;c.height=c.clientHeight*2;}
  rs();addEventListener("resize",rs);
  (function loop(){
    x.clearRect(0,0,c.width,c.height);const W=c.width,H=c.height,mid=H/2;
    x.lineWidth=2;x.strokeStyle="rgba(47,230,166,.85)";x.shadowBlur=10;x.shadowColor="rgba(47,230,166,.6)";
    x.beginPath();
    for(let i=0;i<W;i++){const p=(i/W)*Math.PI*8+t;
      let y=Math.sin(p)*4;const k=(i/W*12+t*1.4)%12;
      if(k>5.4&&k<6.6)y=Math.sin((k-6)*5)*H*0.34;  // ECG spike
      x.lineTo(i,mid-y);}
    x.stroke();x.shadowBlur=0;t+=0.05;requestAnimationFrame(loop);
  })();
})();

/* ---------------- dashboard ---------------- */
function countUp(){
  $$("#stats .stat").forEach((el,i)=>{
    const s=DATA.stats[i];const node=el.querySelector(".v");const dur=900;const t0=performance.now();
    function step(t){let p=Math.min(1,(t-t0)/dur);p=1-Math.pow(1-p,3);
      const val=s.v*p;node.textContent=(s.dec?val.toFixed(s.dec):Math.round(val))+(s.suf||"");
      if(p<1)requestAnimationFrame(step);}
    requestAnimationFrame(step);
  });
}
function renderStats(){
  $("#stats").innerHTML=DATA.stats.map((s,i)=>`
    <div class="stat" style="--vc:${s.vc};--sc:${s.sc||'transparent'};animation-delay:${i*70}ms">
      <div class="k">${s.k}</div><div class="v">0</div><div class="n">${s.d}</div></div>`).join("");
}
function renderRigor(){
  $("#rigor").innerHTML=DATA.rigor.map(r=>`
    <div class="rigor-row ${r.good?'good':'warn'}">
      <div class="rv">${r.v}</div>
      <div style="flex:1"><div class="rl">${r.l}</div><div class="rn">${r.n}</div>
        <div class="meter" style="margin-top:7px"><b style="--w:${r.w}"></b></div></div>
    </div>`).join("");
}
function renderBooks(){
  $("#books").innerHTML=DATA.books.map(b=>`
    <div class="book">
      <div><div class="bn">${b.n}</div><div class="bnote">${b.note}</div></div>
      <div class="stat-mini"><span>SHARPE</span><b>${b.s.toFixed(2)}</b></div>
      <div class="stat-mini"><span>CAGR</span><b>${b.c.toFixed(1)}%</b></div>
      <div class="stat-mini"><span>MAX DD</span><b class="${b.dd<-25?'down':'up'}">${b.dd.toFixed(1)}%</b></div>
      <span class="pill ${b.tag}">${b.tag==='ok'?'PASS':'BENCH'}</span>
    </div>`).join("");
}
function renderEdge(){
  $("#edge").innerHTML=`
    <div class="ehead"><span>Overnight premium</span><b class="up">0.84 ⟶ 0.65</b></div>
    <div class="ebar"><i style="width:71%;background:linear-gradient(90deg,var(--cyan),var(--up))"></i>
      <i style="width:29%;background:var(--line2)"></i></div>
    <div class="erow"><span>return lives at night</span><span class="cyan">ON 0.84 · ID 0.18</span></div>
    <div class="erow"><span>persistence (4 eras)</span><span class="up">not decayed</span></div>
    <div class="erow"><span>drawdown vs buy-hold</span><span class="up">−29% vs −54%</span></div>
    <div class="verdict">A <b>structural</b> risk premium arbitrage can't erase — net of MOC/MOO cost it's
      market-Sharpe with <b>half the drawdown</b>. Real, validated, not alpha: an overnight-beta sleeve.</div>`;
}
function drawEquity(){
  const svg=$("#equity");const W=svg.clientWidth||720,H=svg.clientHeight||212;
  svg.setAttribute("viewBox",`0 0 ${W} ${H}`);
  const pad={l:8,r:8,t:14,b:18};
  const cum=a=>{let v=100,o=[100];a.forEach(r=>{v*=1+r/100;o.push(v);});return o;};
  const S=cum(DATA.strat),P=cum(DATA.spy);
  const max=Math.max(...S,...P),min=Math.min(...S,...P,100);
  const X=i=>pad.l+(W-pad.l-pad.r)*i/(S.length-1);
  const Y=v=>pad.t+(H-pad.t-pad.b)*(1-(v-min)/(max-min));
  const line=arr=>arr.map((v,i)=>(i?"L":"M")+X(i).toFixed(1)+" "+Y(v).toFixed(1)).join(" ");
  const area=arr=>line(arr)+` L${X(arr.length-1)} ${H-pad.b} L${X(0)} ${H-pad.b} Z`;
  // gridlines
  let grid="";for(let g=0;g<4;g++){const y=pad.t+(H-pad.t-pad.b)*g/3;
    grid+=`<line x1="${pad.l}" y1="${y}" x2="${W-pad.r}" y2="${y}" stroke="var(--line)" stroke-width="1"/>`;}
  svg.innerHTML=`
    <defs>
      <linearGradient id="gS" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="rgba(47,230,166,.28)"/><stop offset="1" stop-color="rgba(47,230,166,0)"/></linearGradient>
    </defs>
    ${grid}
    <path d="${area(S)}" fill="url(#gS)"/>
    <path d="${line(P)}" fill="none" stroke="var(--bench)" stroke-width="1.6" stroke-dasharray="3 4" opacity=".85"/>
    <path id="eqline" d="${line(S)}" fill="none" stroke="var(--up)" stroke-width="2.4"
      stroke-linejoin="round" filter="drop-shadow(0 0 6px var(--glow))"/>
    <circle cx="${X(S.length-1)}" cy="${Y(S[S.length-1])}" r="3.5" fill="var(--up)"/>`;
  const path=$("#eqline");const len=path.getTotalLength();
  path.style.strokeDasharray=len;path.style.strokeDashoffset=len;
  path.animate([{strokeDashoffset:len},{strokeDashoffset:0}],{duration:1500,easing:"ease-out",fill:"forwards"});
  $("#chartFoot").innerHTML=`<span>${DATA.years[0]}</span><span>$100k → $${Math.round(S.at(-1)*1000).toLocaleString()} · final ×${(S.at(-1)/100).toFixed(2)} vs SPY ×${(P.at(-1)/100).toFixed(2)}</span><span>${DATA.years.at(-1)}</span>`;
}

/* ---------------- agents field (canvas) ---------------- */
function startField(){
  const c=$("#field"),x=c.getContext("2d");let raf,DPR=Math.min(2,devicePixelRatio||1);
  let nodes=[],edges=[],pulses=[],wave=-1,t=0;
  function layout(){
    c.width=c.clientWidth*DPR;c.height=c.clientHeight*DPR;
    const W=c.width,H=c.height;nodes=[];
    LAYERS.forEach((L,li)=>{const xx=W*(0.14+0.24*li);
      for(let r=0;r<3;r++){const yy=H*(0.26+0.24*r);
        nodes.push({x:xx,y:yy,li,c:COL[L[0]],ph:Math.random()*6.28});}});
    edges=[];for(let li=0;li<3;li++)for(let a=0;a<3;a++)for(let b=0;b<3;b++)
      edges.push([li*3+a,(li+1)*3+b]);
  }
  layout();addEventListener("resize",layout);
  function spawn(n=1){for(let i=0;i<n;i++){const e=edges[(Math.random()*edges.length)|0];
    nodes[e[0]];pulses.push({e,t:Math.random()*0.15,sp:0.006+Math.random()*0.01,c:nodes[e[0]].c});}}
  let spawnAcc=0;
  function frame(){
    const W=c.width,H=c.height;x.clearRect(0,0,W,H);
    // edges
    x.lineWidth=DPR;edges.forEach(([a,b])=>{const n1=nodes[a],n2=nodes[b];
      x.strokeStyle="rgba(120,150,170,.06)";x.beginPath();x.moveTo(n1.x,n1.y);
      const mx=(n1.x+n2.x)/2;x.bezierCurveTo(mx,n1.y,mx,n2.y,n2.x,n2.y);x.stroke();});
    // wave sweep
    if(wave>=0){wave+=0.012;const wx=wave*W;
      const g=x.createLinearGradient(wx-60*DPR,0,wx+60*DPR,0);
      g.addColorStop(0,"rgba(47,230,166,0)");g.addColorStop(.5,"rgba(47,230,166,.5)");g.addColorStop(1,"rgba(47,230,166,0)");
      x.fillStyle=g;x.fillRect(wx-60*DPR,0,120*DPR,H);
      nodes.forEach(n=>{if(Math.abs(n.x-wx)<40*DPR)n.flash=1;});
      if(wave>1.1)wave=-1;}
    // pulses
    spawnAcc+=1;if(spawnAcc>7){spawn(1);spawnAcc=0;}
    pulses=pulses.filter(p=>p.t<1);
    pulses.forEach(p=>{p.t+=p.sp;const n1=nodes[p.e[0]],n2=nodes[p.e[1]];
      const mx=(n1.x+n2.x)/2,u=p.t,iu=1-u;
      const px=iu*iu*n1.x+2*iu*u*mx+u*u*n2.x, py=iu*iu*n1.y+2*iu*u*((n1.y+n2.y)/2)+u*u*n2.y;
      if(p.t>0.96)nodes[p.e[1]].flash=1;
      x.beginPath();x.fillStyle=p.c;x.shadowBlur=12;x.shadowColor=p.c;
      x.arc(px,py,2.4*DPR,0,6.28);x.fill();x.shadowBlur=0;});
    // nodes
    nodes.forEach(n=>{const pulse=1+Math.sin(t*1.6+n.ph)*0.12;
      const r=(n.flash?8.5:6.2)*DPR*pulse;
      x.beginPath();x.fillStyle=n.c;x.shadowBlur=(n.flash?26:14)*DPR;x.shadowColor=n.c;
      x.globalAlpha=n.flash?1:0.92;x.arc(n.x,n.y,r,0,6.28);x.fill();
      x.globalAlpha=1;x.shadowBlur=0;
      x.beginPath();x.fillStyle="#05121a";x.arc(n.x,n.y,r*0.45,0,6.28);x.fill();
      if(n.flash)n.flash*=0.9;if(n.flash<0.05)n.flash=0;});
    t+=0.016;raf=requestAnimationFrame(frame);
  }
  frame();
  const cm=$("#checkMarket");const handler=()=>{wave=0;spawn(14);
    $$(".agent").forEach((a,i)=>setTimeout(()=>{a.classList.add("firing");
      setTimeout(()=>a.classList.remove("firing"),650);},i*45));};
  cm.onclick=handler;
  // periodic auto-check
  const iv=setInterval(handler,9000);setTimeout(handler,600);
  fieldStop=()=>{cancelAnimationFrame(raf);clearInterval(iv);};
}
function renderRoster(){
  $("#roster").innerHTML=LAYERS.map(([key,col,label])=>`
    <div class="rcol" style="--cl:${col}">
      <h4>${label}</h4>
      ${DATA.agents[key].map(([n,r])=>`
        <div class="agent ready" style="--cl:${col}">
          <span class="ad"></span><div><div class="an">${n}</div><div class="ar">${r}</div></div></div>`).join("")}
    </div>`).join("");
}

/* ---------------- control / orchestration ---------------- */
const PHASES=[["research","R","RESEARCH"],["build","B","BUILD"],["validate","V","VALIDATE"],["execute","D","DEPLOY"]];
const SCRIPT=[
  ["research","tag-r","research_agent · scanning 503 S&P names + cross-asset feeds"],
  ["research","tag-r","autonomous_agent · proposing 3 candidate mechanisms"],
  ["research","tag-r","ml_research_agent · framing as triple-barrier classification"],
  ["build","tag-b","code_agent · compiling signals() … validation-retry 1/3"],
  ["build","tag-b","code_agent · module compiled ✓ no look-ahead asserted","ok"],
  ["build","tag-b","ml_code_agent · purged K-fold + embargo wired"],
  ["validate","tag-v","backtesting_agent · walk-forward 5 folds · 2016→2026"],
  ["validate","tag-v","deflated_sharpe · DSR 99% · PBO 52% · reality-check p=0.83"],
  ["validate","tag-v","risk_agent · GATE Sharpe 0.97 ✓ · maxDD −22.8% ✓ · 5/5 folds ✓","ok"],
  ["execute","tag-e","execution_agent · dry-run reconcile vs live book"],
  ["execute","tag-e","execution_agent · plan: SELL BIL 36,853 → BUY SPY 36,996 (2 orders)"],
  ["execute","tag-e","⏸ AWAITING HUMAN AUTHORIZATION — no order sent to broker","warn"],
];
function renderPhases(){
  $("#phaseTrack").innerHTML=PHASES.map(([k,i,l])=>`
    <div class="phase" data-k="${k}" style="--c:${COL[k]}"><span class="pi">${i}</span><span>${l}</span></div>`).join("");
}
let running=false, BACKEND=false;
const TAGOF={research:"tag-r",build:"tag-b",validate:"tag-v",execute:"tag-e"};
const esc=s=>String(s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
function appendLine(ts,tag,txt,cls){
  const log=$("#log");const ln=document.createElement("div");ln.className="ln";
  ln.innerHTML=`<span class="ts">${ts||new Date().toLocaleTimeString("en-US",{hour12:false})}</span>`+
    `<span class="${tag||''}">▌</span><span class="${cls||''}">${esc(txt)}</span>`;
  log.appendChild(ln);log.scrollTop=log.scrollHeight;
}
function setPhase(prev,next){
  if(prev&&prev!==next)$(`.phase[data-k="${prev}"]`)?.classList.replace("run","done");
  if(next)$(`.phase[data-k="${next}"]`)?.classList.add("run");
}
function finishRun(label){
  running=false;$("#power").classList.remove("armed");
  $("#powerLabel").textContent=label||"LOOP COMPLETE";$("#logState").textContent="gated";
  $$(".phase.run").forEach(p=>p.classList.remove("run"));
}
function orchestrate(){
  if(running)return;running=true;
  $("#power").classList.add("armed");$("#powerLabel").textContent="RUNNING…";
  $("#log").innerHTML="";$$(".phase").forEach(p=>p.classList.remove("run","done"));
  BACKEND?realRun():simulateRun();
}
function realRun(){
  $("#logState").textContent="live · backend";
  fetch("/api/run",{method:"POST"}).then(r=>r.json()).then(()=>{
    let since=0,cur=null;
    const poll=()=>fetch("/api/log?since="+since).then(r=>r.json()).then(d=>{
      d.lines.forEach(L=>{since++;
        if(L.phase!==cur){setPhase(cur,L.phase);cur=L.phase;}
        appendLine(L.ts,TAGOF[L.phase],L.text,L.cls);});
      if(d.status==="done"){setPhase(cur,null);finishRun("LOOP COMPLETE · GATED");}
      else setTimeout(poll,350);
    }).catch(()=>{appendLine("","err","[backend connection lost]","err");finishRun("DISCONNECTED");});
    poll();
  }).catch(()=>{BACKEND=false;simulateRun();});
}
function simulateRun(){
  $("#logState").textContent="live · sim";let i=0,cur=null;
  (function next(){
    if(i>=SCRIPT.length){setPhase(cur,null);finishRun("LOOP COMPLETE · GATED");return;}
    const [ph,tag,txt,cls]=SCRIPT[i];
    if(ph!==cur){setPhase(cur,ph);cur=ph;}
    appendLine(new Date().toLocaleTimeString("en-US",{hour12:false}),tag,txt,cls);
    i++;setTimeout(next, txt.includes("AWAITING")?900:420+Math.random()*340);
  })();
}
$("#power").addEventListener("click",orchestrate);
// detect the backend; if present the Control page runs the REAL pipeline
fetch("/api/health").then(r=>r.ok?r.json():null).then(j=>{
  if(j&&j.ok){BACKEND=true;$("#logState").textContent="backend live";$("#logState").classList.add("up");
    $(".log-empty")&&($(".log-empty").textContent="// backend connected — one click runs the real pipeline");}
}).catch(()=>{});

/* ---------------- init ---------------- */
$("#span").textContent=DATA.span;
renderStats();renderRigor();renderBooks();renderEdge();renderRoster();renderPhases();
activateView(location.hash.slice(1)||"dashboard");
addEventListener("resize",()=>{if($("#view-dashboard").classList.contains("is-active"))drawEquity();});
