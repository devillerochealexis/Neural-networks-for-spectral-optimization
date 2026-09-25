import {loadBundle, SpectralModel} from './model.js';
import {compileExpression} from './expression.js';
import {optimize} from './optimizer.js';

let context=null,model=null,active=false,paused=false,stopped=false;
const send=(type,payload={})=>postMessage({type,...payload});
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function checkpoint() {await sleep(0);while(paused&&!stopped)await sleep(40);return !stopped;}
self.onmessage=async({data})=>{
  try {
    if(data.type==='pause'){paused=true;send('state',{state:'paused'});return;}
    if(data.type==='resume'){paused=false;send('state',{state:'running'});return;}
    if(data.type==='stop'){stopped=true;paused=false;send('state',{state:'stopping'});return;}
    if(data.type==='load') {
      if(active)throw Error('Stop the active run before replacing the model.');
      context=null;model=null;
      send('phase',{message:'Loading saved model…'});
      const response=await fetch(data.url);
      if(!response.ok)throw Error(`Model download failed (${response.status}).`);
      const text=await response.text();
      if(text.length>64*1024*1024)throw Error('Model bundle exceeds 64 MB.');
      const bundle=JSON.parse(text);context=loadBundle(bundle);
      context.source=data.url;
      context.bundleSHA256=self.crypto?.subtle?Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(text))),b=>b.toString(16).padStart(2,'0')).join(''):null;
      model=new SpectralModel(context);
      const disk=model.forward(new Float64Array(context.dimension),false);
      send('loaded',{metadata:{name:bundle.name,id:bundle.id,bundleSHA256:context.bundleSHA256,modes:bundle.modes,
        members:bundle.members.length,activation:bundle.members[0].activation,activations:[...new Set(bundle.members.map(m=>m.activation))],
        dimensions:[context.dimension,...bundle.members[0].layers.map(l=>l.output)],rho:bundle.rho,
        bytes:new TextEncoder().encode(text).length},disk:{mean:Array.from(disk.mean),std:Array.from(disk.std)}});
      return;
    }
    if(data.type==='validate') {
      const functional=compileExpression(data.expression);
      if(model){const values=model.forward(new Float64Array(context.dimension),false).mean.map(v=>v/data.area);functional.evaluate(values);}
      send('validated',{requestId:data.requestId,expression:data.expression,derivatives:functional.derivatives});return;
    }
    if(data.type==='run') {
      if(active)throw Error('An optimization is already running.');
      if(!context||!model)throw Error('Load a model first.');
      active=true;paused=false;stopped=false;send('state',{state:'running'});
      try {
        const result=await optimize(context,model,data.settings,{checkpoint,
          phase:message=>send('phase',{message}),progress:event=>send('progress',event)});
        send('result',{result});
      } finally {active=false;paused=false;send('state',{state:'idle'});}
    }
  } catch(error) {send('error',{message:error.message,requestId:data.requestId,operation:data.type});}
};
