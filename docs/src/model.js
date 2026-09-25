/** Dense network operations with reverse-mode differentiation of their input.
 * Native float64 accumulation; weights are the unmodified exported float32 values.
 */
import {rotate} from './geometry.js';

export function decodeArray(encoded, expectedLength, kind='f32') {
  if (!encoded || encoded.dtype !== kind || typeof encoded.base64 !== 'string') throw Error('Invalid model array encoding.');
  const bytes=Uint8Array.from(atob(encoded.base64),c=>c.charCodeAt(0)), size=kind==='f64'?8:4;
  if(bytes.length!==expectedLength*size) throw Error('Model array size does not match its declared dimensions.');
  const view=new DataView(bytes.buffer), result=new Float64Array(expectedLength);
  for(let i=0;i<expectedLength;i++) result[i]=kind==='f64'?view.getFloat64(i*8,true):view.getFloat32(i*4,true);
  if(!result.every(Number.isFinite)) throw Error('Model arrays must contain only finite values.');
  return result;
}
function validVector(x,n,positive=false) {return Array.isArray(x)&&x.length===n&&x.every(v=>Number.isFinite(v)&&(!positive||v>0));}
export function loadBundle(bundle) {
  if(bundle?.format!=='spectral-shape-simple' || bundle.version!==1) throw Error('Expected a version 1 model-only bundle. Run web_simple/scripts/update_app.py.');
  const m=bundle.modes, d=m*2;
  if(!Number.isInteger(m)||m<1||m>32) throw Error('Unsupported harmonic count.');
  if(!Array.isArray(bundle.eigenvalueIndices)||bundle.eigenvalueIndices.length!==10||new Set(bundle.eigenvalueIndices).size!==10||!bundle.eigenvalueIndices.every(k=>Number.isInteger(k)&&k>=1&&k<=10)) throw Error('The model must have all ten eigenvalue outputs.');
  if(!(bundle.rho>0&&bundle.rho<1)||!validVector(bundle.caps,m,true)) throw Error('Missing strict radial positivity constraints.');
  if(!Array.isArray(bundle.members)||bundle.members.length<1||bundle.members.length>8) throw Error('Expected 1–8 ensemble members.');
  const members=bundle.members.map(member=>{
    if(!['relu','tanh','silu'].includes(member.activation)||!validVector(member.xscale,d,true)||!validVector(member.ymean,10)||!validVector(member.yscale,10,true)) throw Error('Unsupported activation or normalization.');
    if(!Array.isArray(member.layers)||member.layers.length<2||member.layers.length>9) throw Error('Expected 1–8 hidden layers.');
    let previous=d;
    const layers=member.layers.map((layer,i)=>{
      if(layer.input!==previous||!Number.isInteger(layer.output)||layer.output<1||layer.output>2048||(i===member.layers.length-1&&layer.output!==10)) throw Error('Invalid layer dimensions.');
      previous=layer.output;
      return {...layer,weights:decodeArray(layer.weights,layer.input*layer.output),bias:decodeArray(layer.bias,layer.output)};
    });
    return {...member,layers};
  });
  if(bundle.predictionRotations!==12) throw Error('This format requires the saved 24-view prediction rule.');
  if('support' in bundle || 'q' in bundle || 'J' in bundle)throw Error('Model-only bundles must not contain training samples or support data.');
  return {bundle,members,dimension:d,modes:m};
}

function activate(v,kind) {return kind==='relu'?Math.max(0,v):kind==='tanh'?Math.tanh(v):v/(1+Math.exp(-v));}
function activationDerivative(v,kind) {
  if(kind==='relu')return v>0?1:0;
  if(kind==='tanh'){const t=Math.tanh(v);return 1-t*t;}
  const s=1/(1+Math.exp(-v));return s+v*s*(1-s);
}
function forward(member,input,retain) {
  let x=Float64Array.from(input,(v,i)=>v/member.xscale[i]); const tape=[];
  for(let l=0;l<member.layers.length;l++) {
    const layer=member.layers[l], z=new Float64Array(layer.output);
    for(let i=0;i<layer.output;i++) {
      let value=layer.bias[i], offset=i*layer.input;
      for(let j=0;j<layer.input;j++) value+=layer.weights[offset+j]*x[j];
      z[i]=value;
    }
    const hidden=l<member.layers.length-1;
    if(retain)tape.push(z);
    x=hidden?z.map(v=>activate(v,member.activation)):z;
  }
  return {log:x.map((v,i)=>v*member.yscale[i]+member.ymean[i]),tape};
}
function backward(member,tape,logWeights) {
  let g=Float64Array.from(logWeights,(v,i)=>v*member.yscale[i]);
  for(let l=member.layers.length-1;l>=0;l--) {
    const layer=member.layers[l], prev=new Float64Array(layer.input);
    for(let i=0;i<layer.output;i++) {
      const v=g[i]*(l<member.layers.length-1?activationDerivative(tape[l][i],member.activation):1);
      if(v===0)continue;
      for(let j=0,offset=i*layer.input;j<layer.input;j++) prev[j]+=v*layer.weights[offset+j];
    }
    g=prev;
  }
  return g.map((v,i)=>v/member.xscale[i]);
}

export class SpectralModel {
  constructor(context) {Object.assign(this,context);this.columns=Array.from({length:10},(_,k)=>context.bundle.eigenvalueIndices.indexOf(k+1));}
  forward(q,retain=true) {
    const predictions=[], tapes=[];
    for(const member of this.members) {
      const logs=new Float64Array(10), views=[];
      for(let i=0;i<12;i++) for(const reflection of [false,true]) {
        const angle=2*Math.PI*i/12, record=forward(member,rotate(q,angle,reflection),retain);
        for(let k=0;k<10;k++)logs[k]+=record.log[k]/24;
        if(retain)views.push({angle,reflection,tape:record.tape});
      }
      const values=logs.map(Math.exp);
      if(!values.every(v=>Number.isFinite(v)&&v>0)) throw Error('The model returned a nonfinite spectrum.');
      predictions.push(values); tapes.push(views);
    }
    const mean=new Float64Array(10),std=new Float64Array(10);
    for(let k=0;k<10;k++) {
      const col=this.columns[k];
      mean[k]=predictions.reduce((s,p)=>s+p[col],0)/predictions.length;
      std[k]=Math.sqrt(predictions.reduce((s,p)=>s+(p[col]-mean[k])**2,0)/predictions.length);
    }
    return {mean,std,predictions,tapes};
  }
  vjp(record,canonicalWeights) {
    const gradient=new Float64Array(this.dimension),m=this.modes;
    for(let n=0;n<this.members.length;n++) {
      const weights=new Float64Array(10);
      for(let k=0;k<10;k++)weights[this.columns[k]]=canonicalWeights[k]*record.predictions[n][this.columns[k]]/(24*this.members.length);
      for(const view of record.tapes[n]) {
        const g=backward(this.members[n],view.tape,weights);
        for(let j=0;j<m;j++) {
          const co=Math.cos((j+1)*view.angle),si=Math.sin((j+1)*view.angle), a=g[j],b=g[j+m]*(view.reflection?-1:1);
          gradient[j]+=co*a+si*b; gradient[j+m]+=-si*a+co*b;
        }
      }
    }
    if(!gradient.every(Number.isFinite)) throw Error('Nonfinite network input gradient.');
    return gradient;
  }
}
