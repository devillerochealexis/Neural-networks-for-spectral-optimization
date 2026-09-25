import {compileExpression} from './expression.js';
import {validateSettings} from './optimizer.js';
import {drawShape,drawHistory,exportSVG} from './plots.js';

const $=id=>document.getElementById(id), worker=new Worker(new URL('./worker.js',import.meta.url),{type:'module'});
let metadata=null,disk=null,ready=false,busy=false,paused=false,best=null,current=null,result=null,trace=[],activeSettings=null,startTime=0,validationId=0;
let presets=[],saved=[],availableModels=[];
try{saved=JSON.parse(localStorage.getItem('spectral-simple-objectives-v1')||'[]');if(!Array.isArray(saved))saved=[];saved=saved.filter(p=>typeof p?.expression==='string'&&typeof p.name==='string').slice(0,30);}catch{saved=[];}
const format=v=>Number.isFinite(v)?Number(v.toPrecision(7)).toString():'—';
const settings=()=>validateSettings({expression:$('expression').value,area:$('area').value,starts:$('starts').value,steps:$('steps').value,seed:$('seed').value,maximize:$('direction').value==='max'});
function showError(message) {$('error').textContent=message;$('error').hidden=!message;}
function setBusy(value) {busy=value;$('settings').disabled=value||!ready;$('model-select').disabled=value||!availableModels.length;$('run').disabled=value||!ready;$('pause').disabled=!value;$('stop').disabled=!value;}
function resetResult() {best=null;current=null;result=null;trace=[];activeSettings=null;$('best-value').textContent='—';$('current-value').textContent='—';$('iteration-label').textContent='No iterations yet';$('result-summary').textContent='Save the best shape, its coefficients, or the full reproducible run.';for(const id of ['save-json','save-csv','save-svg','save-png'])$(id).disabled=true;render();}
function render() {
  const area=activeSettings?.area||Number($('area').value)||1,modes=metadata?.modes||15;
  drawShape($('shape'),best,current,area,modes);drawHistory($('history'),trace);
  $('area-value').textContent=format(area);
  $('floor-value').textContent=`≥ ${(best?1-best.amplitudeSum:1-(metadata?.rho??.75)).toFixed(4)}`;
  const spectrum=best?.predictedEigenvalues||(disk?disk.mean.map(v=>v/area):[]);
  const spread=best?.ensembleStd||(disk?disk.std.map(v=>v/area):[]);
  for(let k=0;k<10;k++) {const cell=$(`eigen-${k}`);cell.textContent=spectrum.length?format(spectrum[k]):'—';cell.title=spread.length?`Ensemble standard deviation: ${format(spread[k])}`:'';}
  $('ordering').textContent=spectrum.some((v,i)=>i>0&&v<spectrum[i-1])?'Some predicted eigenvalues are out of order. Outputs are shown as learned, without sorting.':'Hover over an eigenvalue to see ensemble spread. This is not a confidence interval.';
}
function derivativePreview() {
  try {
    const f=compileExpression($('expression').value);$('derivatives').replaceChildren();
    f.derivatives.forEach((d,k)=>{const p=document.createElement('p');p.textContent=`∂F/∂x${k+1} = ${d}`;$('derivatives').append(p);});
    $('formula-feedback').textContent='Symbolic derivatives ready.';$('formula-feedback').classList.remove('invalid');return f;
  } catch(error) {$('derivatives').replaceChildren();$('formula-feedback').textContent=error.message;$('formula-feedback').classList.add('invalid');return null;}
}
function fillPresets(selectedExpression=$('expression').value) {
  const select=$('preset');select.replaceChildren();let selected='custom';
  [...presets,...saved].forEach((p,i)=>{const option=document.createElement('option');option.value=String(i);option.textContent=p.name;select.append(option);if(p.expression===selectedExpression)selected=String(i);});
  const custom=document.createElement('option');custom.value='custom';custom.textContent='Custom formula';select.append(custom);select.value=selected;
}
function download(blob,name) {const a=document.createElement('a'),url=URL.createObjectURL(blob);a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}
const filename=extension=>`spectral-shape-${new Date().toISOString().replace(/[:.]/g,'-')}.${extension}`;
function currentRecord() {return result||{format:'spectral-shape-snapshot',version:1,appVersion:'simple-1.0.0',createdAt:new Date().toISOString(),status:$('run-state').textContent.toLowerCase(),model:metadata,settings:activeSettings,selection:'surrogate only; no PDE solve',best,current,trace};}
for(let k=0;k<10;k++){const row=document.createElement('div');row.className='spectral-row';const name=document.createElement('span');name.textContent=`λ${String(k+1).replace(/\d/g,d=>'₀₁₂₃₄₅₆₇₈₉'[d])}`;const value=document.createElement('strong');value.id=`eigen-${k}`;value.textContent='—';row.append(name,value);$('spectrum').append(row);}
worker.onmessage=({data})=>{
  if(data.type==='phase'){$('status').textContent=data.message;return;}
  if(data.type==='loaded') {
    metadata=data.metadata;disk=data.disk;ready=true;resetResult();setBusy(false);
    $('model-name').textContent=metadata.name;$('model-details').textContent=`${metadata.dimensions.join(' → ')} · ${metadata.members} members · ${metadata.activations.join(" / ")} · ${(metadata.bytes/1e6).toFixed(1)} MB`;
    $('model-state').textContent='Ready';$('model-state').classList.add('active');$('run-state').textContent='Ready';$('status').textContent='Model loaded. Choose a function and start exploring.';
    $('constraint-details').textContent=`r(θ) / a₀ ≥ ${(1-metadata.rho).toFixed(2)} at every angle. Harmonic caps stay active.`;
    $('plot-note-detail').textContent=`${metadata.modes} harmonics · positive radial graph`;
    derivativePreview();render();return;
  }
  if(data.type==='validated') {
    if(data.requestId!==validationId)return;
    $('formula-feedback').textContent='Valid formula and derivatives at the starting disk.';$('formula-feedback').classList.remove('invalid');return;
  }
  if(data.type==='state') {
    if(data.state==='idle'){setBusy(false);paused=false;$('pause').textContent='Pause';}
    else if(data.state==='paused'){paused=true;$('pause').textContent='Resume';$('run-state').textContent='Paused';$('status').textContent='Paused. Resume or stop to keep the best candidate.';}
    else if(data.state==='stopping'){$('run-state').textContent='Stopping';$('pause').disabled=true;$('stop').disabled=true;}
    else {setBusy(true);paused=false;$('pause').textContent='Pause';$('run-state').textContent='Running';$('run-state').classList.add('active');}
    return;
  }
  if(data.type==='progress') {
    best=data.best;current=data.current;trace.push(...data.trace);$('best-value').textContent=format(best.F);$('current-value').textContent=format(current.F);
    $('plot-title').textContent=paused?'Search paused':'Following the gradient';
    $('iteration-label').textContent=`Start ${data.start+1}/${activeSettings.starts} · step ${data.iteration}/${activeSettings.steps}`;
    $('status').textContent=`${activeSettings.maximize?'Maximizing':'Minimizing'} F · ${current.source.replaceAll('_',' ')}`;
    $('elapsed').textContent=`${data.elapsedSeconds.toFixed(1)} s`;
    for(const id of ['save-json','save-csv','save-svg','save-png'])$(id).disabled=false;
    render();return;
  }
  if(data.type==='result') {
    result=data.result;best=result.best;current=best;trace=result.trace;$('run-state').textContent=result.status==='stopped'?'Stopped':'Complete';$('run-state').classList.remove('active');$('plot-title').textContent=best?'Your best shape':'Search stopped';
    $('elapsed').textContent=`${result.elapsedSeconds.toFixed(1)} s`;
    if(best) {
      $('best-value').textContent=format(best.F);$('current-value').textContent=format(best.F);
      $('status').textContent=`${result.status==='stopped'?'Stopped':'Finished'} · best predicted F = ${format(best.F)}`;
      $('result-summary').textContent=`${best.source.replaceAll('_',' ')} · ${best.termination.replaceAll('_',' ')} · ${result.runs.length/2} starts completed. All values are predictions.`;
    } else {$('status').textContent='Stopped before a candidate was evaluated.';}
    setBusy(false);render();return;
  }
  if(data.type==='error') {
    if(data.operation==='validate') {
      if(data.requestId!==validationId)return;
      $('formula-feedback').textContent=data.message;$('formula-feedback').classList.add('invalid');
    } else {showError(data.message);$('status').textContent='Check the message above before continuing.';$('run-state').textContent='Error';if(data.operation==='load'){ready=false;$('model-state').textContent='Load failed';}setBusy(false);}
  }
};
worker.onerror=event=>{showError(event.message||'The computation worker could not start. Serve this folder over HTTP(S), rather than opening index.html directly.');setBusy(false);};
$('preset').addEventListener('change',()=>{if($('preset').value!=='custom')$('expression').value=[...presets,...saved][Number($('preset').value)].expression;derivativePreview();});
$('expression').addEventListener('input',()=>{$('preset').value='custom';validationId++;derivativePreview();});
$('validate').addEventListener('click',()=>{const f=derivativePreview();if(f)worker.postMessage({type:'validate',expression:f.source,area:Number($('area').value),requestId:++validationId});});
$('save-formula').addEventListener('click',()=>{
  const f=derivativePreview();if(!f)return;
  if(saved.some(p=>p.expression===f.source)){$('formula-feedback').textContent='This formula is already saved.';return;}
  saved.push({name:`Saved: ${f.source.slice(0,48)}`,expression:f.source});saved=saved.slice(-30);
  try{localStorage.setItem('spectral-simple-objectives-v1',JSON.stringify(saved));fillPresets(f.source);$('formula-feedback').textContent='Saved in this browser.';}catch{$('formula-feedback').textContent='Browser storage is unavailable. The current formula can still be used.';}
});
$('run').addEventListener('click',()=>{
  try {const input=settings();resetResult();activeSettings=input;showError('');startTime=performance.now();setBusy(true);$('plot-title').textContent='Preparing the search';worker.postMessage({type:'run',settings:input});}
  catch(error){showError(error.message);}
});
$('pause').addEventListener('click',()=>worker.postMessage({type:paused?'resume':'pause'}));
$('stop').addEventListener('click',()=>worker.postMessage({type:'stop'}));
$('area').addEventListener('input',()=>{validationId++;if(!busy&&!result)render();});
function beginModelLoad(name) {
  validationId++;$('model-state').classList.remove('active');
  ready=false;metadata=null;disk=null;resetResult();setBusy(true);$('pause').disabled=true;$('stop').disabled=true;showError('');$('model-state').textContent='Loading';$('model-name').textContent=name;
}
$('model-select').addEventListener('change',()=>{
  const selected=availableModels[Number($('model-select').value)];
  if(!selected)return;
  beginModelLoad(selected.name);
  worker.postMessage({type:'load',url:selected.url});
});
$('save-json').addEventListener('click',()=>download(new Blob([JSON.stringify(currentRecord(),null,2)],{type:'application/json'}),filename('json')));
$('save-csv').addEventListener('click',()=>{if(!best)return;const m=metadata.modes,rows=best.coefficients.map((v,i)=>`${i===0?'a0':i<=m?'a'+i:'b'+(i-m)},${v}`);download(new Blob(['coefficient,value\n'+rows.join('\n')+'\n'],{type:'text/csv'}),filename('csv'));});
$('save-svg').addEventListener('click',()=>{if(best)download(new Blob([exportSVG(best,activeSettings.area,metadata.modes)],{type:'image/svg+xml'}),filename('svg'));});
$('save-png').addEventListener('click',async()=>{
  if(!best)return;const url=URL.createObjectURL(new Blob([exportSVG(best,activeSettings.area,metadata.modes)],{type:'image/svg+xml'}));
  const image=new Image();image.onload=()=>{const canvas=document.createElement('canvas');canvas.width=1440;canvas.height=1120;canvas.getContext('2d').drawImage(image,0,0,1440,1120);URL.revokeObjectURL(url);canvas.toBlob(blob=>{if(blob)download(blob,filename('png'));},'image/png');};image.onerror=()=>{URL.revokeObjectURL(url);showError('PNG export failed; SVG and JSON remain available.');};image.src=url;
});
async function initialize() {
  render();
  try {const response=await fetch(new URL('../objectives.json',import.meta.url));if(!response.ok)throw Error();presets=await response.json();presets=presets.filter(p=>typeof p.name==='string'&&typeof p.expression==='string');fillPresets();}
  catch{fillPresets();$('formula-feedback').textContent='Preset file unavailable; custom formulas still work.';}
  try {
    const configURL=new URL('../app-model.json',import.meta.url);
    const response=await fetch(configURL,{cache:'no-store'});
    if(!response.ok)throw Error(`Model configuration unavailable (${response.status}).`);
    const config=await response.json();
    if(!Array.isArray(config.models)||!config.models.length||!config.models.every(m=>typeof m?.name==='string'&&typeof m.url==='string'&&/^models\/[A-Za-z0-9._-]+\.spectral\.json$/.test(m.url)))throw Error('Invalid model list in app-model.json.');
    availableModels=config.models.map(m=>({...m,url:new URL(m.url,configURL).href}));
    $('model-select').replaceChildren();
    availableModels.forEach((m,i)=>{const option=document.createElement('option');option.value=String(i);option.textContent=m.name;$('model-select').append(option);});
    beginModelLoad(availableModels[0].name);
    worker.postMessage({type:'load',url:availableModels[0].url});
  } catch(error) {
    showError(error.message);$('model-state').textContent='Load failed';$('status').textContent='Check app-model.json and reload the page.';setBusy(false);
  }
}
initialize();
