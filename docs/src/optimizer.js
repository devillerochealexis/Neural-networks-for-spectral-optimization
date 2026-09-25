import {compileExpression, DomainError} from './expression.js';
import {amplitudes, coefficients, dot, norm, project} from './geometry.js';

export function validateSettings(settings) {
  const s={expression:settings.expression, area:Number(settings.area), starts:Number(settings.starts),
    steps:Number(settings.steps), seed:Number(settings.seed), maximize:Boolean(settings.maximize)};
  if(!Number.isFinite(s.area)||s.area<1e-8||s.area>1e8)throw Error('Area must be between 10⁻⁸ and 10⁸.');
  if(!Number.isInteger(s.starts)||s.starts<1||s.starts>32)throw Error('Choose 1–32 starts.');
  if(!Number.isInteger(s.steps)||s.steps<0||s.steps>2000)throw Error('Choose 0–2000 steps per start.');
  if(!Number.isInteger(s.seed)||s.seed<0||s.seed>0xffffffff)throw Error('Seed must be an integer between 0 and 4294967295.');
  compileExpression(s.expression); return s;
}
export function randomGenerator(seed) {
  let a=seed>>>0;
  return ()=>{a=(a+0x6D2B79F5)>>>0;let t=a;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;};
}
export function objectiveEvaluator(model, functional, area, scale=1, maximize=false) {
  const sign=maximize?-1:1;
  return q=>{
    const record=model.forward(q), spectrum=record.mean.map(v=>v/area);
    const {value:F,partials}=functional.evaluate(spectrum);
    const gradient=model.vjp(record,partials.map(v=>sign*v/(scale*area)));
    return {F,objective:sign*F/scale,gradient,spectrum,J:record.mean,std:record.std.map(v=>v/area),partials};
  };
}
export function chooseStarts(context, settings) {
  // Geometry and seed only: independent of model weights, spectra and objective.
  const {dimension:d,modes:m}=context,{rho,caps}=context.bundle;
  const random=randomGenerator(settings.seed),starts=[{source:'disk',q:new Float64Array(d)}];
  for(let i=1;i<settings.starts;i++) {
    const target=rho*(.1+.9*random()),q=new Float64Array(d);let total=0;
    for(let j=0;j<m;j++) {
      const amplitude=caps[j]*random(),phase=2*Math.PI*random();total+=amplitude;
      q[j]=amplitude*Math.cos(phase);q[j+m]=amplitude*Math.sin(phase);
    }
    const shrink=Math.min(1,target/Math.max(total,1e-30));
    starts.push({source:`random_${i}`,q:q.map(v=>v*shrink)});
  }
  return starts;
}

export async function optimize(context, model, input, hooks={}) {
  const settings=validateSettings(input), functional=compileExpression(settings.expression);
  const {rho,caps,id,name}=context.bundle, d=context.dimension, m=context.modes;
  const diskValue=objectiveEvaluator(model,functional,settings.area)(new Float64Array(d)).F;
  const scale=Math.max(1,Math.abs(diskValue)),evaluate=objectiveEvaluator(model,functional,settings.area,scale,settings.maximize);
  const now=()=>performance.now(), begin=now(), checkpoint=hooks.checkpoint||(()=>Promise.resolve(true));
  const projectShape=q=>project(q,rho,caps);
  const inputScale=context.members[0].xscale;
  const pairScale=Array.from({length:m},(_,j)=>Math.hypot(inputScale[j],inputScale[j+m])/Math.SQRT2);
  const sorted=pairScale.slice(0,Math.min(3,m)).sort((a,b)=>a-b);
  const median=sorted[Math.floor(sorted.length/2)];
  const preconditioner=Float64Array.from([...pairScale,...pairScale],v=>m>3?(v/median)**2:1);
  let best=null,stopped=false,events=0,lastEmission=-Infinity; const runs=[],trace=[],skippedStarts=[];
  hooks.phase?.('Generating random starting shapes');
  const starts=chooseStarts(context,settings);
  const candidate=(q,value,source,kind,iterations,termination,pg)=>({q:Array.from(q),coefficients:coefficients(q,settings.area),
    F:value.F,objective:value.objective,predictedEigenvalues:Array.from(value.spectrum),predictedJ:Array.from(value.J),
    ensembleStd:Array.from(value.std),functionalPartials:Array.from(value.partials),source,kind,iterations,termination,
    projectedGradientNorm:pg,amplitudeSum:amplitudes(q).reduce((s,v)=>s+v,0)});
  const emit=(current,start,iteration,force=false)=>{
    if(!best||current.objective<best.objective)best=current;
    trace.push({event:events++,start,iteration,F:current.F,bestF:best.F});
    if(force||now()-lastEmission>100) {hooks.progress?.({current,best,start,iteration,trace:trace.slice(-1),elapsedSeconds:(now()-begin)/1000});lastEmission=now();}
  };
  hooks.phase?.('Optimizing');
  for(let s=0;s<starts.length;s++) {
    if(await checkpoint()===false){stopped=true;break;}
    const start=starts[s];let q=projectShape(start.q),value;
    try {value=evaluate(q);}catch(error){if(!(error instanceof DomainError))throw error;skippedStarts.push(start.source);continue;}
    const initialProjection=projectShape(q.map((v,j)=>v-value.gradient[j]));
    let pg=norm(q.map((v,j)=>v-initialProjection[j]));
    const incumbent=candidate(q,value,start.source,'incumbent',0,'incumbent',pg);
    incumbent.startQ=Array.from(start.q);runs.push(incumbent);emit(incumbent,s,0,true);
    let step=.2,iterations=0,termination=settings.steps?'iteration_budget':'incumbent';const history=[];
    for(let it=0;it<settings.steps;it++) {
      if(await checkpoint()===false){stopped=true;termination='stopped';break;}
      const projected=projectShape(q.map((v,j)=>v-value.gradient[j]));pg=norm(q.map((v,j)=>v-projected[j]));
      history.push({iteration:it,F:value.F,objective:value.objective,projectedGradientNorm:pg});iterations++;
      if(pg<2e-5){termination='projected_gradient';break;}
      let accepted=false;
      for(let trial=0;trial<18;trial++) {
        if(await checkpoint()===false){stopped=true;termination='stopped';break;}
        let update=value.gradient.map((v,j)=>step*v*preconditioner[j]);const factor=Math.min(1,.06/Math.max(norm(update),1e-20));update=update.map(v=>v*factor);
        const next=project(q.map((v,j)=>v-update[j]),rho,caps,preconditioner.subarray(0,m));
        let trialValue;
        try {trialValue=evaluate(next);}catch(error){if(!(error instanceof DomainError))throw error;step*=.5;continue;}
        if(trialValue.objective<=value.objective+1e-4*dot(value.gradient,next.map((v,j)=>v-q[j]))) {
          q=next;value=trialValue;step=Math.min(step*1.3,3);accepted=true;
          emit(candidate(q,value,start.source,'gradient_run',iterations,'running',pg),s,it+1);break;
        }
        step*=.5;
      }
      if(stopped)break;
      if(!accepted||step<1e-8){termination=accepted?'step_too_small':'line_search';break;}
    }
    const projected=projectShape(q.map((v,j)=>v-value.gradient[j]));pg=norm(q.map((v,j)=>v-projected[j]));
    const endpoint=candidate(q,value,start.source,'gradient_run',iterations,termination,pg);
    endpoint.history=history;endpoint.startQ=Array.from(start.q);runs.push(endpoint);emit(endpoint,s,iterations,true);
    if(stopped)break;
  }
  // Pick a persisted endpoint/incumbent, rather than a transient progress record.
  best=runs.reduce((win,r)=>!win||r.objective<win.objective?r:win,null);
  if(best) {
    if(best.amplitudeSum>rho+1e-10||amplitudes(best.q).some((a,j)=>a>caps[j]+1e-10))throw Error('Final shape failed its feasibility check.');
  }
  return {format:'spectral-shape-run',version:1,appVersion:'simple-1.0.0',createdAt:new Date().toISOString(),
    status:stopped?'stopped':'complete',selection:'surrogate only; no PDE solve',
    model:{id,name,bundleSHA256:context.bundleSHA256,source:context.source},settings,
    objectiveScale:scale,symbolicGradient:functional.derivatives,area:settings.area,rho,caps,
    relativeRadialFloor:1-rho,trainingDataUsed:false,supportRestriction:false,
    startRng:'mulberry32',startStrategy:'disk + capped random amplitudes and uniform phases (v1)',skippedStarts,
    starts:starts.map(s=>({...s,q:Array.from(s.q)})),best,runs,trace,elapsedSeconds:(now()-begin)/1000};
}
