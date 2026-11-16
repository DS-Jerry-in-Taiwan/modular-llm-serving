'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const terminal = new Set(['completed', 'failed', 'cancelled', 'deadline_exceeded']);
  const phaseNames = {queued:'等待執行', loading:'載入模型', running:'正在推論', generating:'正在推論', cleaning_up:'清理中', completed:'已完成', failed:'執行失敗', cancelled:'已停止', deadline_exceeded:'已超時', checking:'驗證模型'};
  const V2_API = '/api/v2';
  let info = null, job = null, jobTimer = null, infoTimer = null, polling = false, submitting = false;
  let history = [], activeExample = 'custom', currentRequest = null, executionMode = 'cold';
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
  function residentMode() {return executionMode==='resident';}
  function modeLabel() {return residentMode() ? 'Warm resident' : 'Cold benchmark';}
  function renderMode() {
    const resident=residentMode();
    document.querySelectorAll('input[name="execution-mode"]').forEach(input=>{input.checked=input.value===executionMode;});
    document.querySelector('.variant-picker').hidden=resident;
    $('streamOption').hidden=!resident;
    $('streamOutput').disabled=!resident;
    $('protocolBadge').textContent=resident ? 'V2 · resident / contract v1' : 'V1 · cold / JSON';
    $('protocolDetail').textContent=resident ? '常駐 engine 由 FIFO admission queue 管理；串流只接受 V2 StreamEvent。' : '每個配置獨立啟動 llama-server；請求時間不含模型載入。';
    $('modeNotice').textContent=resident ? 'Warm resident 會保留一個模型 engine。請求先進入 bounded queue；可串流收到 chunk，terminal event 才代表完成。' : 'Cold benchmark 會依序執行兩個配置，每次切換需重新載入模型。測試期間請保持頁面開啟；重新整理可恢復查看。';
  }
  function lifecycleValue(value) {return value === undefined || value === null || value === '' ? '—' : String(value);}
  function renderLifecycle() {
    const visible=residentMode() || job?.execution_mode==='resident';
    $('v2Lifecycle').hidden=!visible;
    if(!visible) return;
    const timings=job?.phase_timings || job?.timings || {}, metrics=job?.metrics || {};
    $('lifecycleStatus').textContent=job ? `${modeLabel()} · ${phase(job.status)}` : '等待請求';
    $('lifecycleProtocol').textContent=`contract v${job?.contract_version || 1}`;
    $('requestId').textContent=lifecycleValue(job?.request_id);
    $('runId').textContent=lifecycleValue(job?.run_id || job?.id);
    $('queueState').textContent=lifecycleValue(job?.queue_status || (job?.status==='queued' ? 'queued' : job ? 'admitted' : null));
    $('streamState').textContent=job?.stream ? `${job.stream_status || 'open'}${Number.isInteger(job.stream_sequence) ? ` · ${job.stream_sequence} events` : ''}` : 'non-stream';
    $('terminalState').textContent=job?.terminal_event ? 'received' : job && terminal.has(job.status) ? phase(job.status) : 'pending';
    const timingParts=Object.entries(timings).map(([key,value])=>`${key}: ${fmt(typeof value==='object' ? value.value : value)} s`);
    $('phaseTimings').textContent=timingParts.length ? timingParts.join(' · ') : '尚未收到 phase timing。';
    const metricParts=Object.entries(metrics).map(([key,value])=>`${key}: ${lifecycleValue(typeof value==='object' ? value.value : value)}`);
    $('v2Metrics').textContent=metricParts.length ? metricParts.join(' · ') : '尚未收到 metrics。';
  }
  function updateControls() {
    const active = job && !terminal.has(job.status), chosen = [...document.querySelectorAll('.variant-picker input:checked')];
    $('run').disabled=submitting || !info?.ready || info?.busy || active || !$('prompt').value.trim() || (!residentMode() && chosen.length===0);
    $('run').firstElementChild.textContent=job?.status==='failed' ? '重新執行測試' : residentMode() ? '送出 V2 請求' : chosen.length===2 ? '開始 A/B 測試' : `執行 ${chosen[0]?.value==='optimized' ? 'B 最佳化' : 'A 基準'}配置`;
    $('cancel').hidden=!active; $('cancel').disabled=submitting || job?.status==='cancelling';
    $('export').disabled=!job;
    document.querySelectorAll('.variant-picker input').forEach(el=>el.disabled=!!active || submitting);
    $('prompt').disabled=!!active || submitting; $('maxTokens').disabled=!!active || submitting;
    $('streamOutput').disabled=!residentMode() || !!active || submitting;
    document.querySelectorAll('.tab').forEach(el=>el.disabled=!!active || submitting);
    $('charCount').textContent=$('prompt').value.length.toLocaleString();
    renderMode();
    renderLifecycle();
  }
  function renderInfo() {
    const ready=info?.ready, busy=info?.busy;
    $('connectionDot').className=`status-dot ${ready ? '' : info?.error ? 'error' : 'checking'}`;
    $('connectionTitle').textContent=ready ? busy ? '本機正在執行測試' : '環境已驗證' : info?.error ? '環境尚未就緒' : '正在驗證執行環境';
    $('connectionDetail').textContent=info?.error || (ready ? busy ? '一次執行一組測試，請稍候。' : '開始測試時載入模型。' : info?.status || '驗證模型檔案與執行環境…');
    if(info?.model) $('modelName').textContent=info.model;
    if(Number.isInteger(info?.context_size) && info.context_size>0) $('contextSize').textContent=info.context_size.toLocaleString('zh-TW');
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
    const card=$(`card-${id}`), result=job?.variants?.find(v=>v.id===id); let label=id==='baseline' ? 'A · 基準配置' : 'B · 最佳化配置';
    if(residentMode() && id==='optimized') {card.hidden=true; card.replaceChildren(); return;}
    card.hidden=false;
    if(residentMode() && id==='baseline') label='V2 · 常駐模型';
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
    renderLifecycle();
  }
  function v2Path(id='') {return `${V2_API}/runs${id ? `/${encodeURIComponent(id)}` : ''}`;}
  function applyV2Payload(payload) {
    const events=payload?.events || payload?.stream_events || [];
    let text=typeof payload?.text==='string' ? payload.text : (job?.text || '');
    let sequence=job?.stream_sequence;
    for(const event of events) {
      if(!event || typeof event!=='object') continue;
      if(event.kind==='chunk' && typeof event.text==='string') {text+=event.text;sequence=event.sequence;}
      if(event.kind==='terminal') {payload.terminal_event=true;payload.stream_status='terminal';}
      if(event.kind==='error') {payload.error=event.error?.message || '串流協定回報錯誤';payload.stream_status='error';}
    }
    if(text) payload.text=text;
    if(Number.isInteger(sequence)) payload.stream_sequence=sequence;
    if(payload.stream && !payload.stream_status) payload.stream_status='open';
    payload.variants=Array.isArray(payload.variants) ? payload.variants : [{id:'baseline',status:payload.status || 'running',text:payload.text || '',request_seconds:payload.metrics?.request_seconds,output_tokens:payload.metrics?.completion_tokens}];
    return payload;
  }
  async function pollJob(id) {
    if(polling) return;
    polling=true;
    try {
      const next=await api(residentMode() || job?.execution_mode==='resident' ? v2Path(id) : `/api/runs/${encodeURIComponent(id)}`);
      if(job?.id && job.id!==id) return;
      job=(residentMode() || job?.execution_mode==='resident') ? applyV2Payload({...next,execution_mode:'resident',run_id:next.run_id || id,request_id:next.request_id || job?.request_id}) : next;error('');renderJob();
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
    const resident=residentMode();
    currentRequest=resident ? {contract_version:1,request_id:globalThis.crypto?.randomUUID ? globalThis.crypto.randomUUID() : `req-${Date.now()}`,prompt:$('prompt').value,model_id:info?.model_id || 'qwen',config_id:info?.config_id || 'resident',max_new_tokens:Number($('maxTokens').value),stream:$('streamOutput').checked,idempotency_key:`web-${Date.now()}-${Math.random().toString(16).slice(2)}`} : {prompt:$('prompt').value,max_tokens:Number($('maxTokens').value),variants:[...document.querySelectorAll('.variant-picker input:checked')].map(el=>el.value)};
    try {
      const next=await api(resident ? v2Path() : '/api/runs',{method:'POST',body:JSON.stringify(currentRequest)});
      const id=next.id || next.run_id || next.request_id;
      if(!id) throw new Error('伺服器未回傳 run_id');
      clearTimeout(jobTimer);job=resident ? applyV2Payload({...next,id,execution_mode:'resident',status:next.status || 'queued',request_id:next.request_id || currentRequest.request_id,run_id:next.run_id || id,stream:currentRequest.stream,variants:[]}) : {id,status:'queued',variants:currentRequest.variants.map(id=>({id,status:'queued'}))};
      storage.set('qwen-lab-job',id);storage.set('qwen-lab-input',JSON.stringify({...currentRequest,execution_mode:resident?'resident':'cold'}));
      renderJob();await pollJob(id);
    } catch(err) {error(`無法開始測試：${err.message}。請檢查服務狀態後再次執行。`);await refreshInfo();}
    finally {submitting=false;updateControls();}
  }
  async function cancel() {
    if(!job || terminal.has(job.status)) return;
    $('cancel').disabled=true;
    try {await api(residentMode() || job.execution_mode==='resident' ? `${v2Path(job.id)}/cancel` : `/api/runs/${encodeURIComponent(job.id)}/cancel`,{method:'POST',body:'{}'});$('jobStatus').textContent='正在停止並清理模型…';}
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
  async function loadReference() {
    const box=$('referenceData');
    try {
      const data=await api('/api/reference');
      if(data.label) $('referenceTitle').textContent=data.label;
      if(data.scope) $('referenceScope').textContent=data.scope;
      box.replaceChildren();
      if(data.status!=='available') {box.append(make('p','reference-empty','目前沒有可用的 V1 固定測試基準。你仍可在上方執行即時測試。'));return;}
      const totals=data.measured_request_totals_seconds || {}, gpu=data.gpu_peak_bytes || {};
      const summary=make('div','reference-metrics');
      const total=make('div','reference-stat');total.append(make('span','',`${Number.isFinite(data.quality?.samples_per_configuration)?data.quality.samples_per_configuration+' 次':''}請求時間合計（不含載入）`),make('strong','',`A ${fmt(totals.baseline)} s → B ${fmt(totals.candidate)} s`));
      const ratio=make('div','reference-stat');ratio.append(make('span','','A/B 請求時間比'),make('strong','',`${fmt(totals.latency_ratio)}×`));
      const memory=make('div','reference-stat');memory.append(make('span','','整張 GPU 記憶體峰值'),make('strong','',`A ${Number.isFinite(gpu.baseline)?(gpu.baseline/1e9).toFixed(2):'—'} / B ${Number.isFinite(gpu.candidate)?(gpu.candidate/1e9).toFixed(2):'—'} GB`));
      summary.append(total,ratio,memory);box.append(summary);
      if(Array.isArray(data.cases) && data.cases.length) {
        const wrap=make('div','reference-table-wrap'),table=make('table','reference-table'),caption=make('caption','','六個固定案例 · 每案兩次請求的平均時間');table.append(caption);
        const head=make('thead'),header=make('tr');
        ['固定案例','A 基準（秒）','B 最佳化（秒）','A/B 時間比'].forEach(label=>{const th=make('th','',label);th.scope='col';header.append(th);});head.append(header);table.append(head);
        const body=make('tbody');
        data.cases.forEach(item=>{const row=make('tr'),name=make('th','',item.label || item.id);name.scope='row';row.append(name,make('td','',fmt(item.baseline_seconds)),make('td','',fmt(item.candidate_seconds)),make('td','',`${fmt(item.latency_ratio)}×`));body.append(row);});
        table.append(body);wrap.append(table);box.append(wrap);
      }
      const quality=data.quality || {}, qualityNote=make('p','reference-quality');
      const passed=quality.baseline==='pass' && quality.candidate==='pass';
      qualityNote.append(make('span',`badge ${passed?'completed':''}`,passed?'固定樣本品質檢查通過':'品質檢查狀態未確認'),document.createTextNode(` ${quality.review || '請查看報告中的品質審閱紀錄'}${Number.isFinite(quality.samples_per_configuration)?` · 每種配置 ${quality.samples_per_configuration} 個樣本`:''}。此結果不適用於上方的自訂回答。`));box.append(qualityNote);
      if(Array.isArray(data.limitations) && data.limitations.length) {const list=make('ul','reference-limitations');data.limitations.forEach(item=>list.append(make('li','',item)));box.append(list);}
      const ids=make('p','reference-ids');ids.append(make('span','',`A 紀錄：${data.baseline_run_id || '—'}`),make('span','',`B 紀錄：${data.candidate_run_id || '—'}`));box.append(ids);
    } catch {box.replaceChildren(make('p','reference-empty','暫時無法載入固定測試基準。即時測試仍可使用；完整資料請查看固定測試報告。'));}
  }
  $('run').addEventListener('click',run);$('cancel').addEventListener('click',cancel);
  document.querySelectorAll('input[name="execution-mode"]').forEach(input=>input.addEventListener('change',()=>{
    if(job && !terminal.has(job.status)) return;
    executionMode=input.value;job=null;clearTimeout(jobTimer);error('');renderMode();renderJob();
  }));
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
    try {const input=JSON.parse(storage.get('qwen-lab-input') || 'null');if(input) {$('prompt').value=input.prompt || ''; $('maxTokens').value=String(input.max_tokens || input.max_new_tokens || 192); executionMode=input.execution_mode==='resident' ? 'resident' : 'cold';}} catch {}
    job={id:remembered,status:'queued',execution_mode:executionMode,variants:[]};renderJob();pollJob(remembered);
  }
  renderMode();refreshInfo();refreshHistory();loadReference();
})();
