/* ============================================================
   QUANT·DESK — app.js  (no deps, file:// safe)
   ============================================================ */
const DATA = {
  span: "2016 — 2026 · adjusted daily",
  // real yearly returns of OUR equity ensemble book vs SPY (build_ensemble())
  years: [2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026],
  spy:   [13.6,21.7,-4.6,31.2,18.3,28.7,-18.2,26.2,24.9,17.7,11.0],
  strat: [3.1,12.3,1.8,11.7,16.7,14.2,-3.8,15.9,17.2,13.5,8.5],
  stats: [
    {k:"Sharpe",        v:1.59,  d:"vs SPY 0.82",        vc:"var(--up)",  dec:2, sc:"rgba(47,230,166,.12)"},
    {k:"CAGR",          v:10.5,  d:"equity ensemble",    vc:"var(--text)",suf:"%",dec:1},
    {k:"Max Drawdown",  v:-7.8,  d:"vs SPY −33.7%",      vc:"var(--up)",  suf:"%",dec:1},
    {k:"Walk-forward",  v:5,     d:"of 5 folds positive",vc:"var(--text)",suf:"/5"},
    {k:"Unit tests",    v:76,    d:"rigor · engine · CV",vc:"var(--cyan)",suf:" ✓"},
  ],
  rigor: [
    {l:"Deflated Sharpe (best sleeve)", v:"99.4%", w:"99%", good:1, n:"corrected for 13 trials"},
    {l:"Prob. of Backtest Overfitting", v:"52%",   w:"52%", good:0, n:"single-sleeve pick ≈ noise"},
    {l:"Reality-Check vs SPY (p)",      v:"0.83",  w:"83%", good:0, n:"no alpha over passive"},
    {l:"Overnight edge · net Sharpe",   v:"0.65",  w:"65%", good:1, n:"½ drawdown · DSR 99% · 5/5 WF"},
  ],
  books: [
    {n:"Equity ensemble", note:"no-margin · crash sentinel · the book we run", s:1.59, c:10.5, dd:-7.8, tag:"ok"},
  ],
  // offline seed of web/book.json (the live registry when the backend is up)
  book: {name:"Equity ensemble (no-margin, crash sentinel)", strategies:[
    {name:"rsi2_meanrev",label:"RSI-2 mean reversion",weight:0.252,source:"core",family:"reversion"},
    {name:"donchian",label:"Donchian breakout",weight:0.198,source:"core",family:"trend"},
    {name:"recovery",label:"Recovery thrust (V-snapback)",weight:0.162,source:"core",family:"trend"},
    {name:"trend_5020",label:"50/200 trend",weight:0.126,source:"core",family:"trend"},
    {name:"lowvol_defensive",label:"Low-vol defensive",weight:0.10,source:"core",family:"structure"},
    {name:"pead",label:"Post-earnings drift",weight:0.09,source:"core",family:"structure"},
    {name:"xs_momentum",label:"Cross-sectional dual-momentum",weight:0.072,source:"core",family:"trend"},
    {name:"mean_gravity",label:"gravity (lab)",weight:0.05,source:"lab",family:"reversion",validated:true}
  ], overlay:"VIX crash sentinel — de-risk to 60% on early-warning or VIX backwardation"},
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
const FAMC = {reversion:"var(--cyan)",trend:"var(--up)",volatility:"var(--bench)",structure:"var(--violet)"};

/* offline fallback: a real snapshot of runners/agent_lab.py --emit web/candidates.json
   (used when the page is opened from file:// with no backend). With the backend up
   the cockpit fetches the LIVE /api/candidates instead. */
const DEMO_BENCH={"sharpe":1.591,"cagr":0.1051,"maxdd":-0.0775};
const DEMO_SRSTAR=0.674;
const DEMO_CANDS=[{"agent":"gravity","strategy":"mean_gravity","family":"reversion","thesis":"stretch below a long anchor in ATR units mean-reverts to the anchor","sharpe":0.785,"maxdd":-0.0371,"corr":0.217,"blend":1.612,"delta":0.0216,"wf_pos":5,"wf_n":5,"dsr":0.6415,"verdict":"REVIEW","reason":"improves the blend but DSR 64%"},{"agent":"regime-dial","strategy":"vol_regime_switch","family":"volatility","thesis":"size inversely to the volatility percentile - full when calm, flat when stormy","sharpe":1.366,"maxdd":-0.0557,"corr":0.808,"blend":1.597,"delta":0.006,"wf_pos":5,"wf_n":5,"dsr":0.9852,"verdict":"REVIEW","reason":"improves the blend but corr too high"},{"agent":"breadth-int","strategy":"breadth_thrust_self","family":"trend","thesis":"a name's own internal breadth thrust signals broad-based ignition","sharpe":0.0,"maxdd":0.0,"corr":0.0,"blend":1.591,"delta":-0.0,"wf_pos":0,"wf_n":5,"dsr":0.015,"verdict":"REJECT","reason":"the ensemble already owns this mechanism"},{"agent":"ladder-keeper","strategy":"drawdown_ladder","family":"reversion","thesis":"accumulate deeper-into-the-dip in a secular uptrend, scale out on recovery","sharpe":0.893,"maxdd":-0.0934,"corr":0.574,"blend":1.582,"delta":-0.009,"wf_pos":5,"wf_n":5,"dsr":0.7596,"verdict":"REJECT","reason":"the ensemble already owns this mechanism"},{"agent":"coil-scout","strategy":"coil_release","family":"volatility","thesis":"compressed ranges store energy that releases upward - trade the transition","sharpe":1.14,"maxdd":-0.1632,"corr":0.761,"blend":1.57,"delta":-0.0205,"wf_pos":5,"wf_n":5,"dsr":0.9283,"verdict":"REJECT","reason":"the ensemble already owns this mechanism"},{"agent":"nightfade","strategy":"gap_fade_revert","family":"reversion","thesis":"panic gap-down opens in an uptrend overshoot and snap back","sharpe":0.432,"maxdd":-0.0683,"corr":0.395,"blend":1.568,"delta":-0.0229,"wf_pos":3,"wf_n":5,"dsr":0.227,"verdict":"REJECT","reason":"the ensemble already owns this mechanism"},{"agent":"two-clocks","strategy":"dual_horizon_agree","family":"structure","thesis":"act only when a slow uptrend and a fast turn-up agree","sharpe":0.668,"maxdd":-0.0877,"corr":0.489,"blend":1.561,"delta":-0.03,"wf_pos":5,"wf_n":5,"dsr":0.492,"verdict":"REJECT","reason":"the ensemble already owns this mechanism"},{"agent":"pathwise","strategy":"trend_persistence","family":"trend","thesis":"the smoothness of a climb, not its slope, predicts continuation","sharpe":0.673,"maxdd":-0.0636,"corr":0.601,"blend":1.557,"delta":-0.0334,"wf_pos":5,"wf_n":5,"dsr":0.4984,"verdict":"REJECT","reason":"the ensemble already owns this mechanism"},{"agent":"streakwatch","strategy":"streak_reversal","family":"reversion","thesis":"rare consecutive down-streaks in an uptrend are short-term overreactions","sharpe":0.326,"maxdd":-0.0673,"corr":0.432,"blend":1.555,"delta":-0.0354,"wf_pos":4,"wf_n":5,"dsr":0.1331,"verdict":"REJECT","reason":"the ensemble already owns this mechanism"},{"agent":"straightline","strategy":"slope_quality","family":"trend","thesis":"grade exposure by the R-squared of the up-trend - buy clean, refuse ragged","sharpe":0.635,"maxdd":-0.0671,"corr":0.681,"blend":1.548,"delta":-0.043,"wf_pos":5,"wf_n":5,"dsr":0.451,"verdict":"REJECT","reason":"the ensemble already owns this mechanism"},{"agent":"inflection","strategy":"velocity_flip","family":"trend","thesis":"price acceleration turns before the trend cross - buy the inflection","sharpe":0.963,"maxdd":-0.0974,"corr":0.75,"blend":1.542,"delta":-0.0487,"wf_pos":5,"wf_n":5,"dsr":0.8189,"verdict":"REJECT","reason":"the ensemble already owns this mechanism"},{"agent":"ignition","strategy":"expansion_breakout","family":"volatility","thesis":"a true-range blow-out closing strong ignites a multi-day move","sharpe":0.101,"maxdd":-0.1355,"corr":0.466,"blend":1.522,"delta":-0.0682,"wf_pos":2,"wf_n":5,"dsr":0.0337,"verdict":"REJECT","reason":"the ensemble already owns this mechanism"}];

/* offline fallback for the RAG-Vault sentiment panel (used file:// or vault down) */
const DEMO_SIGNALS={ok:true,as_of:"2026-06-22",universe:50,demo:true,
  longs:[{ticker:"NVDA",conviction:0.60,strength:0.54,confidence:"high"},
         {ticker:"AVGO",conviction:0.49,strength:0.45,confidence:"medium"},
         {ticker:"AMD",conviction:0.41,strength:0.39,confidence:"medium"}],
  shorts:[{ticker:"MSFT",conviction:-0.55,strength:0.50,confidence:"high"},
          {ticker:"IBM",conviction:-0.43,strength:0.40,confidence:"medium"},
          {ticker:"HPQ",conviction:-0.38,strength:0.36,confidence:"medium"}]};

const $ = (s,r=document)=>r.querySelector(s);
const $$ = (s,r=document)=>[...r.querySelectorAll(s)];

/* ---------------- routing (with #hash deep-links) ---------------- */
let fieldStop=null;
function activateView(v){
  const b=$(`.nav-item[data-view="${v}"]`);if(!b)v="dashboard";
  $$(".nav-item").forEach(n=>n.classList.toggle("is-active",n.dataset.view===v));
  $$(".view").forEach(x=>x.classList.remove("is-active"));
  $("#view-"+v).classList.add("is-active");
  if(v==="dashboard"){drawEquity();countUp();pollAccount();loadBook();loadSignals();}
  if(v==="agents"){startField();loadCandidates();} else if(fieldStop){fieldStop();fieldStop=null;}
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
function renderBook(book){
  const el=$("#bookList");if(!el)return;
  const ss=(book.strategies||[]).slice().sort((a,b)=>(b.weight||0)-(a.weight||0));
  el.innerHTML=ss.map(s=>{
    const w=(s.weight!=null&&s.weight>0)?(s.weight*100).toFixed(1)+"%":"new";
    const lab=s.source==="lab";
    return `<div class="bk-row${lab?' lab':''}" style="--fc:${FAMC[s.family]||'var(--dim2)'}">
      <span class="bk-dot"></span>
      <div class="bk-name">${s.label||s.name}<span>${s.name}</span></div>
      <span class="bk-src">${lab?(s.validated?'lab · validated':'lab · approved'):'core'}</span>
      <b class="bk-w">${w}</b></div>`;
  }).join("")+(book.overlay?`<div class="bk-overlay">⛨ ${book.overlay}</div>`:"");
}
function loadBook(){
  if(!BACKEND){renderBook(DATA.book);return;}
  fetch("/api/book").then(r=>r.json()).then(b=>renderBook(b&&b.strategies?b:DATA.book)).catch(()=>renderBook(DATA.book));
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
/* ---- RAG-Vault sentiment signals (live LONG/SHORT verdicts) ---- */
function sigCol(title,rows,dir){
  const cls=dir==="long"?"up":"down",arrow=dir==="long"?"▲":"▼";
  const body=rows.length?rows.map(r=>`
    <div class="sig-row">
      <b class="sig-tk">${esc2(r.ticker)}</b>
      <span class="sig-conf ${esc2(r.confidence)}">${esc2(r.confidence)}</span>
      <div class="sig-meter"><i class="${cls}" style="width:${Math.round(Math.min(1,r.strength)*100)}%"></i></div>
      <b class="sig-cv ${cls}">${r.conviction>=0?'+':''}${r.conviction.toFixed(2)}σ</b>
    </div>`).join(""):`<div class="sig-empty">none today</div>`;
  return `<div class="sig-col"><div class="sig-col-h ${cls}">${arrow} ${title}<span>${rows.length}</span></div>${body}</div>`;
}
function renderSignals(d){
  const tag=$("#sigTag"),grid=$("#sigGrid"),foot=$("#sigFoot");if(!grid)return;
  if(!d||!d.ok){
    tag.textContent="vault offline";tag.className="tag amber";
    grid.innerHTML=`<div class="sig-off">RAG Vault not reachable at <code>${esc2((d&&d.url)||'127.0.0.1:8000')}</code>.
      Start it — <code>uvicorn sp500_vault.api:app --port 8000</code> — to stream live verdicts.</div>`;
    foot.innerHTML="";return;}
  tag.textContent=(d.demo?"demo · ":"")+`as_of ${d.as_of||'—'} · ${d.universe} names`;
  tag.className="tag "+(d.demo?"":"up");
  grid.innerHTML=sigCol("LONG",d.longs||[],"long")+sigCol("SHORT",d.shorts||[],"short");
  foot.innerHTML=`IC-weighted blend — Claude sentiment + supplier lead-lag + 8-K event drift.
    Drive the live book with <code>--sentiment-overlay</code> (gate · opt-in · fail-safe): trade the longs, drop the shorts, concentrate into the matches; flat / uncovered names are left to the algorithm.`;
}
function loadSignals(){
  if(!BACKEND){renderSignals(DEMO_SIGNALS);return;}
  fetch("/api/signals").then(r=>r.json()).then(d=>renderSignals(d&&d.ok?d:(d||{ok:false})))
    .catch(()=>renderSignals(DEMO_SIGNALS));
}
/* ---- live book feed (real Alpaca data via backend) ---- */
const money=n=>"$"+Math.round(n).toLocaleString();
function renderLive(d){
  if(!d||!d.accounts){return;}
  $("#livePanel").hidden=false;
  const spy=d.spy_today;
  $("#liveTs").textContent="updated "+d.ts+(spy!=null?` · SPY ${spy>=0?'+':''}${(spy*100).toFixed(2)}%`:"");
  let tot=0,totPl=0,any=false;
  const cards=d.accounts.map((a,i)=>{
    if(a.status!=="ok"){
      return `<div class="live-card off" style="animation-delay:${i*60}ms"><div class="lc-name">${a.name}<span>#${a.id}</span></div>
        <div class="lc-eq" style="font-size:15px;color:var(--dim2)">${a.status==='no-keys'?'no keys':'offline'}</div></div>`;}
    any=true;tot+=a.equity;totPl+=a.pl;
    const up=a.today>=0,vs=spy!=null?a.today-spy:null;
    const holds=(a.top||[]).map(p=>{
      const v=p.vdir, vc=v==='long'?'up':'down';
      const ttl=`${p.sym} · ${money(p.mv)}`+(v?` · vault ${v} ${p.vconv>=0?'+':''}${p.vconv}σ (${p.vconf})`:' · vault: flat / not covered');
      const pill=v?`<i class="vp ${vc}">${v==='long'?'▲':'▼'}</i>`:'';
      return `<span class="lc-hold${v?' '+vc:''}" title="${esc2(ttl)}">${esc2(p.sym)}${pill}</span>`;
    }).join("");
    return `<div class="live-card" style="animation-delay:${(i+1)*60}ms">
      <div class="lc-name">${a.name}<span>#${a.id} · ${a.n_pos} pos</span></div>
      <div class="lc-eq">${money(a.equity)}</div>
      <div class="lc-row"><span class="${up?'up':'down'}">${up?'▲':'▼'} ${(a.today*100).toFixed(2)}%</span>
        ${vs!=null?`<span class="lc-vs ${vs>=0?'up':'down'}">${vs>=0?'+':''}${(vs*100).toFixed(2)}% vs SPY</span>`:''}</div>
      ${a.top&&a.top.length?`<div class="lc-holds" title="vault verdict per held name">${holds}</div>`:''}</div>`;
  }).join("");
  const tUp=totPl>=0;
  const totalCard=`<div class="live-card total">
    <div class="lc-name">Total book<span>${d.accounts.filter(a=>a.status==='ok').length}/3 live</span></div>
    <div class="lc-eq">${any?money(tot):'—'}</div>
    <div class="lc-row"><span class="${tUp?'up':'down'}">${tUp?'▲':'▼'} ${money(Math.abs(totPl))} today</span></div></div>`;
  $("#liveGrid").innerHTML=totalCard+cards;
  renderSentActions(d.sentiment);
}
/* gate preview: which held names --sentiment-overlay would TRADE (long) vs DROP (short) */
function renderSentActions(s){
  const el=$("#sentActions");if(!el)return;
  if(!s||!s.ok||(!s.long.length&&!s.short.length)){el.innerHTML="";return;}
  const chips=(arr,cls,arrow)=>arr.length?arr.map(r=>
    `<span class="sa-chip ${cls}" title="${esc2(r.sym)} · vault ${r.conv>=0?'+':''}${r.conv}σ (${esc2(r.conf)})">${arrow} ${esc2(r.sym)}</span>`
    ).join(""):`<span class="sa-none">none today</span>`;
  el.innerHTML=`<div class="sa-head">sentiment overlay · <b class="up">gate</b>
      <span>what <code>--sentiment-overlay</code> trades · as_of ${s.as_of||'—'}</span></div>
    <div class="sa-row"><b class="up">▲ TRADE · go long (${s.long.length})</b><div class="sa-chips">${chips(s.long,'up','▲')}</div></div>
    <div class="sa-row"><b class="down">✕ DROP · don't trade (${s.short.length})</b><div class="sa-chips">${chips(s.short,'down','✕')}</div></div>
    <div class="sa-foot">Capital freed by the drops is concentrated into the long-confirmed names; flat / uncovered names are left to the algorithm.</div>`;
}
function pollAccount(){
  if(!BACKEND)return;
  fetch("/api/account").then(r=>r.json()).then(renderLive).catch(()=>{});
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
  // let the lab pipeline flash a whole layer as each phase fires
  window.__pulseLayer=(key)=>{const li=LAYERS.findIndex(L=>L[0]===key);if(li<0)return;
    for(let r=0;r<3;r++){const nn=nodes[li*3+r];if(nn)nn.flash=1;}
    if(li<3)spawn(2);};
  // periodic auto-check
  const iv=setInterval(handler,9000);setTimeout(handler,600);
  fieldStop=()=>{cancelAnimationFrame(raf);clearInterval(iv);window.__pulseLayer=null;};
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

/* ---------------- strategy lab (12 agents → candidates → approve/reject) ---------------- */
const esc2=s=>String(s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
function agentLine(L){
  const log=$("#agentLog");if(!log)return;
  const tok=((L.text||"").split(" | ")[0]||"").trim();
  const known=["research","build","validate","execute","verdict"].includes(tok);
  const lk=tok==="verdict"?"execute":tok;
  const rest=known?L.text.slice(L.text.indexOf("|")+1):L.text;
  const ln=document.createElement("div");ln.className="aln";
  ln.innerHTML=`<span class="ats">${L.ts||""}</span>`+
    (known?`<span class="aph" style="--c:${COL[lk]}">${tok}</span>`:`<span class="aph dim"></span>`)+
    `<span class="${L.cls||''}">${esc2(rest)}</span>`;
  log.appendChild(ln);log.scrollTop=log.scrollHeight;
  if(known&&window.__pulseLayer)window.__pulseLayer(lk);
}
function finishAgents(){
  running=false;const b=$("#runAgents");if(b){b.disabled=false;b.classList.remove("busy");}
  const s=$("#agentsState");if(s){s.textContent="complete · your decision";s.className="tag up";}
}
function renderCandidates(data){
  const bm=data.benchmark||DEMO_BENCH,dec=data.decisions||{};
  $("#benchLine").textContent=`benchmark — equity ensemble · Sharpe ${bm.sharpe.toFixed(2)} · the bar to clear`;
  const cands=(data.candidates||[]).slice().sort((a,b)=>b.delta-a.delta);
  const n=cands.length, imp=cands.filter(c=>c.delta>0).length;
  const modeTxt=data.mode==='llm'?`<b class="violet">${data.n_llm||0} Claude-invented</b>`
    :(data.mode==='param-search'?'<b class="amber">param-search (LLM offline)</b>':'demo batch');
  $("#candGrid").innerHTML=
    `<div class="lab-summary">${modeTxt} · batch #${data.seed||'demo'} · ${n} mechanisms ·
      <b class="up">${imp}</b> raise the blend ·
      <b class="amber">${cands.filter(c=>c.verdict==='REVIEW').length}</b> for review ·
      SR* ${(data.sr_star_annual||DEMO_SRSTAR).toFixed(2)} (deflated for ${n} trials)</div>`+
    cands.map(c=>{
      const vc=c.verdict==="PROMOTE"?"up":c.verdict==="REVIEW"?"amber":"down";
      const d=dec[c.strategy]?dec[c.strategy].decision:"pending";
      const up=c.delta>=0, lc=c.corr<0.5;
      return `<div class="cand ${d}" data-strat="${esc2(c.strategy)}" style="--fc:${FAMC[c.family]||'var(--dim2)'}">
        <div class="cand-head">
          <div><div class="cand-agent">${esc2(c.agent)}</div><div class="cand-strat">${esc2(c.strategy)}${c.source?`<span class="src-badge ${c.source}">${c.source==='llm'?'LLM-invented':'param-search'}</span>`:''}</div></div>
          <span class="vpill ${vc}">${c.verdict}</span></div>
        <div class="cand-thesis">${esc2(c.thesis)}</div>
        ${c.params?`<div class="cand-params">${esc2(Object.entries(c.params).map(([k,v])=>k+' '+v).join(' · '))}</div>`:''}
        <div class="cand-metrics">
          <div><span>SHARPE</span><b>${c.sharpe.toFixed(2)}</b></div>
          <div><span>CORR</span><b class="${lc?'up':''}">${c.corr>=0?'+':''}${c.corr.toFixed(2)}</b></div>
          <div><span>BLEND Δ</span><b class="${up?'up':'down'}">${up?'+':''}${c.delta.toFixed(2)}</b></div>
          <div title="${c.wf_folds?'walk-forward fold Sharpes: '+c.wf_folds.join(', '):'walk-forward folds'}"><span>WF</span><b>${c.wf_pos}/${c.wf_n}</b></div>
          <div><span>DSR</span><b>${Math.round(c.dsr*100)}%</b></div>
          <div><span>MAX DD</span><b class="down">${(c.maxdd*100).toFixed(1)}%</b></div>
        </div>
        <div class="cand-foot">
          <button class="appr" data-d="approved">✓ approve</button>
          <button class="rej" data-d="rejected">✕ reject</button>
          <span class="cand-state">${d!=='pending'?d:''}</span>
        </div></div>`;
    }).join("");
  $$("#candGrid .cand").forEach(card=>card.querySelectorAll("button").forEach(
    btn=>btn.addEventListener("click",()=>decide(card.dataset.strat,btn.dataset.d,card))));
}
function decide(strategy,decision,card){
  const apply=()=>{card.classList.remove("approved","rejected","pending");card.classList.add(decision);
    card.querySelector(".cand-state").textContent=decision==="approved"?"✓ in book":decision;
    loadBook();};
  if(!BACKEND){
    DATA.book.strategies=DATA.book.strategies.filter(s=>s.name!==strategy);
    if(decision==="approved"){const c=DEMO_CANDS.find(x=>x.strategy===strategy)||{};
      DATA.book.strategies.push({name:strategy,label:c.agent||strategy,weight:0,source:"lab",
        family:c.family,sharpe:c.sharpe,corr:c.corr,delta:c.delta});}
    apply();return;
  }
  fetch("/api/decide",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({strategy,decision})}).then(r=>r.json()).then(apply).catch(apply);
}
function loadCandidates(){
  if(!BACKEND){if(!$("#candGrid").children.length)
    renderCandidates({benchmark:DEMO_BENCH,sr_star_annual:DEMO_SRSTAR,candidates:DEMO_CANDS,decisions:{}});return;}
  fetch("/api/candidates").then(r=>r.json()).then(d=>{
    if(d&&d.candidates&&d.candidates.length)renderCandidates(d);
  }).catch(()=>{});
}
function simAgents(){
  let i=0;const script=DEMO_CANDS.flatMap(c=>[
    `research | ${c.agent} | hypothesis: ${c.thesis}`,
    `build | ${c.agent} | compiled signal | long/flat | shift=1 (no look-ahead)`,
    `validate | ${c.agent} | Sharpe ${c.sharpe.toFixed(2)} | corr->ens ${c.corr>=0?'+':''}${c.corr.toFixed(2)} | blend ${c.blend.toFixed(2)} (${c.delta>=0?'+':''}${c.delta.toFixed(2)}) | WF ${c.wf_pos}/5 | DSR ${Math.round(c.dsr*100)}%`,
    `verdict | ${c.agent} | ${c.verdict} - ${c.reason}`]);
  (function next(){
    if(i>=script.length){agentLine({ts:"",text:"⏸ AWAITING HUMAN DECISION — approve or reject below",cls:"warn"});
      finishAgents();renderCandidates({benchmark:DEMO_BENCH,sr_star_annual:DEMO_SRSTAR,candidates:DEMO_CANDS,decisions:{}});return;}
    agentLine({ts:new Date().toLocaleTimeString("en-US",{hour12:false}),text:script[i],cls:""});
    i++;setTimeout(next,90+Math.random()*110);
  })();
}
function runAgents(){
  if(running)return;running=true;
  const b=$("#runAgents");if(b){b.disabled=true;b.classList.add("busy");}
  const s=$("#agentsState");s.className="tag amber";
  const log=$("#agentLog");log.hidden=false;log.innerHTML="";
  if(!BACKEND){
    s.textContent="demo — start backend for live Claude invention";
    $("#candGrid").innerHTML=`<div class="lab-summary">Offline demo. Run <b>python web/server.py</b> and open it at 127.0.0.1:8787 so the agents invent live via Claude.</div>`;
    simAgents();return;
  }
  s.textContent="Claude inventing strategies… (~1–2 min)";
  $("#candGrid").innerHTML=`<div class="lab-summary">⟳ Claude is inventing a fresh batch and validating each one against the ensemble — watch the log above. This takes ~1–2 minutes.</div>`;
  fetch("/api/run",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({mode:"agents"})})
    .then(r=>r.json()).then(()=>{
      let since=0;
      const poll=()=>fetch("/api/log?since="+since).then(r=>r.json()).then(d=>{
        d.lines.forEach(L=>{since++;agentLine(L);});
        if(d.status==="done"){finishAgents();loadCandidates();}
        else setTimeout(poll,300);
      }).catch(()=>{agentLine({ts:"",text:"[backend connection lost]",cls:"err"});finishAgents();});
      poll();
    }).catch(()=>{BACKEND=false;simAgents();});
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
$("#runAgents")&&$("#runAgents").addEventListener("click",runAgents);
// detect the backend; if present the Control page runs the REAL pipeline
fetch("/api/health").then(r=>r.ok?r.json():null).then(j=>{
  if(j&&j.ok){BACKEND=true;$("#logState").textContent="backend live";$("#logState").classList.add("up");
    $(".log-empty")&&($(".log-empty").textContent="// backend connected — one click runs the real pipeline");
    pollAccount();setInterval(pollAccount,25000);loadBook();}
}).catch(()=>{});

/* ---------------- init ---------------- */
$("#span").textContent=DATA.span;
renderStats();renderRigor();renderBooks();renderEdge();renderRoster();renderPhases();
activateView(location.hash.slice(1)||"dashboard");
addEventListener("resize",()=>{if($("#view-dashboard").classList.contains("is-active"))drawEquity();});
