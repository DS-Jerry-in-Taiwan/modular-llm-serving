'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const terminal = new Set(['completed', 'failed', 'cancelled']);
  const phaseNames = {queued:'等待執行', loading:'載入模型', running:'正在推論', completed:'已完成', failed:'執行失敗', cancelled:'已停止', checking:'驗證模型'};
  let info = null, job = null, jobTimer = null, infoTimer = null, polling = false, submitting = false;
  let history = [], activeExample = 'custom', currentRequest = null;
  const storage = {get(key) {try {return localStorage.getItem(key);} catch {return null;}}, set(key,value) {try {localStorage.setItem(key,value);} catch {}}, remove(key) {try {localStorage.removeItem(key);} catch {}}};
  const make = (tag, cls, value) => {const el = document.createElement(tag); if(cls) el.className=cls; if(value!==undefined) el.textContent=value; return el;};
  const fmt = value => Number.isFinite(value) ? value.toFixed(2) : '—';
  const phase = value => phaseNames[value] || value || '等待執行';
  function error(message) {$('formError').textContent=message || ''; $('formError').hidden=!message;}
  async function api(path, options={}) {
    const response = await fetch(path, {...options, headers:{'Content-Type':'application/json', ...(options.headers || {})}, cache:'no-store'});
    let data; try {data=await response.json();} catch {throw new Error(`伺服器回應格式錯誤（${response.status}）`);}
    if(!response.ok) {const failure=new Error(typeof data.error==='string' ? data.error : typeof data.detail==='string' ? data.detail : `請求失敗（${response.status}）`);failure.status=response.status;throw failure;}
    return data;
  }
  function updateControls() {
    const active = job && !terminal.has(job.status), chosen = [...document.querySelectorAll('.variant-picker input:checked')];
    $('run').disabled=submitting || !info?.ready || info?.busy || active || !$('prompt').value.trim() || chosen.length===0;
    $('run').firstElementChild.textContent=job?.status==='failed' ? '重新執行測試' : chosen.length===2 ? '開始 A/B 測試' : `執行 ${chosen[0]?.value==='optimized' ? 'B 最佳化' : 'A 基準'}配置`;
    $('cancel').hidden=!active; $('cancel').disabled=submitting || job?.status==='cancelling';
    $('export').disabled=!job;
    document.querySelectorAll('.variant-picker input').forEach(el=>el.disabled=!!active || submitting);
    $('prompt').disabled=!!active || submitting; $('maxTokens').disabled=!!active || submitting;
    document.querySelectorAll('.tab').forEach(el=>el.disabled=!!active || submitting);
    $('charCount').textContent=$('prompt').value.length.toLocaleString();
  }
  function renderInfo() {
    const ready=info?.ready, busy=info?.busy;
    $('connectionDot').className=`status-dot ${ready ? '' : info?.error ? 'error' : 'checking'}`;
    $('connectionTitle').textContent=ready ? busy ? '本機正在執行測試' : '模型已就緒' : info?.error ? '環境尚未就緒' : '正在檢查本機模型';
    $('connectionDetail').textContent=info?.error || (ready ? busy ? '一次執行一組測試，請稍候。' : '已驗證模型，可以開始測試。' : info?.status || '驗證模型檔案與執行環境…');
    if(info?.model) $('modelName').textContent=info.model;
    updateControls();
  }
  async function refreshInfo() {
    clearTimeout(infoTimer);
    try {
      info=await api('/api/info');
      const limits=info.limits || {};
      $('prompt').maxLength=limits.max_prompt_chars || 3000;
      $('charLimit').textContent=$('prompt').maxLength.toLocaleString();
      $('maxTokens').querySelectorAll('option').forEach(option => option.disabled=Number(option.value)<(limits.min_tokens || 16) || Number(option.value)>(limits.max_tokens || 256));
      renderExamples(); renderInfo();
    } catch(err) {
      info={ready:false,error:err.message}; renderInfo();
      $('connectionTitle').textContent='無法連線至本機服務';
    }
    infoTimer=setTimeout(refreshInfo, info?.ready && !info?.busy ? 10000 : 1800);
  }
  function renderExamples() {
    const tabs=$('examples');
    if(tabs.dataset.ready) return;
    if(!info?.examples?.length) return;
    for(const example of info.examples) {
      const button=make('button','tab',example.label); button.dataset.example=example.id;
      button.addEventListener('click',()=>{if(job && !terminal.has(job.status)) return; $('prompt').value=example.prompt; activeExample=example.id; selectTab(); updateControls();});
      tabs.append(button);
    }
    tabs.dataset.ready='true';
  }
  function selectTab() {document.querySelectorAll('.tab').forEach(el=>{const active=el.dataset.example===activeExample;el.classList.toggle('active',active);el.setAttribute('aria-pressed',String(active));});}
  function metric(parent, value, unit, label) {
    const box=make('div','metric'), number=make('strong','',value);
    number.append(make('span','',unit)); box.append(number,make('small','',label)); parent.append(box);
  }
  function renderCard(id) {
    const card=$(`card-${id}`), result=job?.variants?.find(v=>v.id===id), label=id==='baseline' ? 'A · 基準配置' : 'B · 最佳化配置';
    card.replaceChildren();
    const head=make('div','answer-head'), heading=make('div','answer-heading');
    heading.append(make('span','config-letter',id==='baseline'?'A':'B'),make('span','',label.slice(4)));
    const state=result?.status || (job ? '未選取' : '等待測試');
    head.append(heading,make('span',`badge ${result?.status || ''}`,phase(state)));card.append(head);
    const content=make('div','answer-content');
    if(result?.text) content.append(make('pre','answer-text',result.text));
    else {
      const empty=make('div','empty-state');
      const active=result && ['loading','running'].includes(result.status);
      empty.append(make('span',active ? 'spinner' : 'empty-symbol',active ? '' : '↳'));
      let text='執行測試後，回答會出現在這裡';
      if(result) text=result.status==='loading' ? '正在載入此配置的模型…' : result.status==='running' ? '模型正在產生回答，完成後一次顯示。' : result.status==='queued' ? '等待前一個配置完成後執行' : result.status==='failed' ? result.error || '推論失敗，請稍後重試。' : result.status==='cancelled' ? '此配置的測試已停止。' : '模型未回傳回答內容。';
      else if(job) text='這次測試未選取此配置';
      empty.append(make('span','',text)); content.append(empty);
    }
    if(result?.text && result.error) content.append(make('p','error-banner',result.error));
    if(result?.truncated) content.append(make('p','badge failed','回答已截斷 · 已達長度上限，內容可能不完整'));
    card.append(content);
    const metrics=make('div','metrics');
    metric(metrics,fmt(result?.request_seconds),'s','請求時間（不含載入）');
    metric(metrics,fmt(result?.tokens_per_second),'tok/s','生成速度');
    metric(metrics,Number.isFinite(result?.gpu_peak_bytes) ? (result.gpu_peak_bytes/1e9).toFixed(2) : '—','GB','整張 GPU 記憶體峰值');
    card.append(metrics);
    const foot=make('div','answer-foot');
    foot.append(make('span','',`模型載入 ${fmt(result?.load_seconds)} s · 輸出 ${Number.isFinite(result?.output_tokens) ? result.output_tokens : '—'} tokens`));
    const copy=make('button','copy','複製回答');copy.disabled=!result?.text;
    copy.addEventListener('click',async()=>{try {await navigator.clipboard.writeText(result.text);copy.textContent='已複製 ✓';setTimeout(()=>copy.textContent='複製回答',1600);} catch {error('瀏覽器無法存取剪貼簿，請直接選取回答文字複製。');}});
    foot.append(copy);card.append(foot);
  }
  function renderJob() {
    renderCard('baseline'); renderCard('optimized'); updateControls();
    $('jobStatus').textContent=job ? `${phase(job.status)}${job.current_variant && !terminal.has(job.status) ? ` · ${job.current_variant==='baseline'?'A 基準':'B 最佳化'} · ${phase(job.phase)}` : ''}` : '等待開始';
    const compare=$('comparison'); compare.hidden=true;compare.replaceChildren();
    const a=job?.variants?.find(v=>v.id==='baseline'),b=job?.variants?.find(v=>v.id==='optimized');
    if(job?.status==='completed' && a?.status==='completed' && b?.status==='completed' && !a.truncated && !b.truncated && a.request_seconds>0 && b.request_seconds>0) {
      compare.hidden=false;
      const ratio=a.request_seconds/b.request_seconds, change=(1-b.request_seconds/a.request_seconds)*100;
      compare.append(make('strong','',`B 請求時間${change>=0?'減少':'增加'} ${Math.abs(change).toFixed(1)}% · A/B 時間比 ${ratio.toFixed(2)}×`),make('span','', '僅比較本次相同輸入；輸出長度可能不同，時間比不代表品質保證。'));
    }
    if(job?.error) error(job.error);
  }
  async function pollJob(id) {
    if(polling) return;
    polling=true;
    try {
      const next=await api(`/api/runs/${encodeURIComponent(id)}`);
      if(job?.id && job.id!==id) return;
      job=next;error('');renderJob();
      if(terminal.has(job.status)) {clearTimeout(jobTimer);await Promise.allSettled([refreshHistory(),refreshInfo()]);}
      else jobTimer=setTimeout(()=>pollJob(id),900);
    } catch(err) {
      if(err.status===404) {clearTimeout(jobTimer);storage.remove('qwen-lab-job');job=null;renderJob();error('這筆測試紀錄已不存在（服務可能已重新啟動）。可以重新執行測試。');await refreshInfo();return;}
      error(`讀取測試進度失敗：${err.message}。將自動重新連線。`);
      jobTimer=setTimeout(()=>pollJob(id),2500);
    } finally {polling=false;}
  }
  async function run() {
    if($('run').disabled) return;
    error(''); submitting=true;updateControls();
    currentRequest={prompt:$('prompt').value,max_tokens:Number($('maxTokens').value),variants:[...document.querySelectorAll('.variant-picker input:checked')].map(el=>el.value)};
    try {
      const next=await api('/api/runs',{method:'POST',body:JSON.stringify(currentRequest)});
      clearTimeout(jobTimer);job={id:next.id,status:'queued',variants:currentRequest.variants.map(id=>({id,status:'queued'}))};
      storage.set('qwen-lab-job',next.id);storage.set('qwen-lab-input',JSON.stringify(currentRequest));
      renderJob();await pollJob(next.id);
    } catch(err) {error(`無法開始測試：${err.message}。請檢查服務狀態後再次執行。`);await refreshInfo();}
    finally {submitting=false;updateControls();}
  }
  async function cancel() {
    if(!job || terminal.has(job.status)) return;
    $('cancel').disabled=true;
    try {await api(`/api/runs/${encodeURIComponent(job.id)}/cancel`,{method:'POST',body:'{}'});$('jobStatus').textContent='正在停止並清理模型…';}
    catch(err) {error(`停止失敗：${err.message}`);$('cancel').disabled=false;}
  }
  async function refreshHistory() {
    try {
      const data=await api('/api/history');history=(data.runs || []).slice(0,20);
      const box=$('history');box.replaceChildren();
      if(!history.length) box.append(make('p','history-empty','還沒有測試紀錄。從上方輸入第一道題開始。'));
      for(const item of history) {
        const row=make('button','history-row'),when=new Date(item.created_at);
        row.append(make('span','history-time',Number.isNaN(when.getTime()) ? '' : when.toLocaleString('zh-TW',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'})),make('span','history-prompt',item.prompt || `測試 ${item.id}`),make('span',`badge ${item.status}`,phase(item.status)),make('span','','↗'));
        row.disabled=!!job && !terminal.has(job.status);
        row.addEventListener('click',()=>loadHistory(item));box.append(row);
      }
    } catch(err) {$('history').replaceChildren(make('p','history-empty',`無法載入紀錄：${err.message}。可按「重新整理」重試。`));}
  }
  async function loadHistory(item) {
    if(job && !terminal.has(job.status)) return;
    clearTimeout(jobTimer);error('');job=item;storage.set('qwen-lab-job',item.id);
    if(item.prompt) {$('prompt').value=item.prompt;activeExample='custom';selectTab();}
    if(item.max_tokens) $('maxTokens').value=String(item.max_tokens);
    renderJob();await pollJob(item.id);$('comparison').parentElement.scrollIntoView({behavior:'smooth',block:'start'});
  }
  $('run').addEventListener('click',run);$('cancel').addEventListener('click',cancel);
  $('prompt').addEventListener('input',()=>{activeExample='custom';selectTab();updateControls();});
  $('maxTokens').addEventListener('change',updateControls);
  document.querySelectorAll('.variant-picker input').forEach(el=>el.addEventListener('change',updateControls));
  $('examples').firstElementChild.addEventListener('click',()=>{activeExample='custom';selectTab();$('prompt').focus();});
  $('refreshInfo').addEventListener('click',refreshInfo);$('refreshHistory').addEventListener('click',refreshHistory);
  $('export').addEventListener('click',()=>{
    if(!job) return;
    const blob=new Blob([JSON.stringify(job,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),link=make('a');
    link.href=url;link.download=`qwen-test-${job.id}.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  renderJob();selectTab();
  const remembered=storage.get('qwen-lab-job');
  if(remembered) {
    try {const input=JSON.parse(storage.get('qwen-lab-input') || 'null');if(input) {$('prompt').value=input.prompt || ''; $('maxTokens').value=String(input.max_tokens || 192);}} catch {}
    job={id:remembered,status:'queued',variants:[]};renderJob();pollJob(remembered);
  }
  refreshInfo();refreshHistory();
})();
