import {boundary,coefficients} from './geometry.js';
const NS='http://www.w3.org/2000/svg';
const element=(name,attrs={},text)=>{const e=document.createElementNS(NS,name);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e;};
export function drawShape(svg, best, current, area, modes, forExport=false) {
  svg.replaceChildren();const width=720,height=560,pad=52;
  svg.setAttribute('xmlns',NS);svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
  svg.append(element('rect',{width,height,fill:'white'}));
  const disk=boundary(coefficients(new Float64Array(2*modes),area)), bp=best?boundary(best.coefficients):[],cp=current?boundary(current.coefficients):[];
  const extent=Math.max(...[...disk,...bp,...cp].flat().map(Math.abs))*1.16;
  const scale=Math.min(width-2*pad,height-2*pad)/(2*extent), sx=x=>width/2+x*scale,sy=y=>height/2-y*scale;
  const tick=10**Math.floor(Math.log10(extent)), spacing=extent/tick<2?tick/2:tick;
  for(let t=Math.ceil(-extent/spacing)*spacing;t<=extent;t+=spacing) {
    svg.append(element('line',{x1:sx(t),y1:sy(extent),x2:sx(t),y2:sy(-extent),stroke:Math.abs(t)<spacing/10?'#d3dfe2':'#edf2f3','stroke-width':1}));
    svg.append(element('line',{x1:sx(-extent),y1:sy(t),x2:sx(extent),y2:sy(t),stroke:Math.abs(t)<spacing/10?'#d3dfe2':'#edf2f3','stroke-width':1}));
    svg.append(element('text',{x:sx(t),y:sy(-extent)+18,'text-anchor':'middle',fill:'#83979f','font-size':10,'font-family':'system-ui'},Number(t.toPrecision(3)).toString()));
  }
  const path=(points,attrs)=>svg.append(element('path',{d:points.map(([x,y],i)=>`${i?'L':'M'}${sx(x).toFixed(3)},${sy(y).toFixed(3)}`).join(' ')+' Z',...attrs}));
  path(disk,{fill:'none',stroke:'#a9b8bf','stroke-width':1.7,'stroke-dasharray':'6 5'});
  if(cp.length&&!forExport)path(cp,{fill:'none',stroke:'#668fae','stroke-width':1.8,opacity:.8});
  if(bp.length)path(bp,{fill:'#087f7610',stroke:'#087f76','stroke-width':2.6});
  if(forExport) {
    svg.append(element('text',{x:24,y:28,fill:'#142e3b','font-size':15,'font-family':'system-ui'},`Spectral Shape Lab · predicted F = ${best.F.toPrecision(8)}`));
    svg.append(element('text',{x:24,y:height-15,fill:'#607782','font-size':11,'font-family':'system-ui'},`Area ${area} · ${modes} harmonics · neural prediction, no PDE solve`));
  }
}
export function drawHistory(svg, trace) {
  svg.replaceChildren();const w=620,h=240,p={left:55,right:16,top:15,bottom:33};
  if(!trace.length) {svg.append(element('text',{x:w/2,y:h/2,'text-anchor':'middle',fill:'#92a3aa','font-size':13,'font-family':'system-ui'},'The optimization path will appear here.'));return;}
  let lo=Math.min(...trace.map(t=>Math.min(t.F,t.bestF))),hi=Math.max(...trace.map(t=>Math.max(t.F,t.bestF)));
  const margin=Math.max((hi-lo)*.1,Math.abs(hi)*.0001,1e-8);lo-=margin;hi+=margin;
  const sx=i=>p.left+i/Math.max(trace.length-1,1)*(w-p.left-p.right),sy=v=>h-p.bottom-(v-lo)/(hi-lo)*(h-p.top-p.bottom);
  for(let k=0;k<4;k++) {const v=lo+(hi-lo)*k/3,y=sy(v);svg.append(element('line',{x1:p.left,x2:w-p.right,y1:y,y2:y,stroke:'#e7edef'}));svg.append(element('text',{x:p.left-9,y:y+3,'text-anchor':'end',fill:'#7b919a','font-size':10,'font-family':'system-ui'},Number(v.toPrecision(5)).toString()));}
  const draw=(key,color,width)=>svg.append(element('path',{d:trace.map((r,i)=>`${i?'L':'M'}${sx(i)},${sy(r[key])}`).join(' '),fill:'none',stroke:color,'stroke-width':width,'stroke-linejoin':'round'}));
  draw('F','#87a5bb',1.5);draw('bestF','#087f76',2.3);
  svg.append(element('text',{x:p.left,y:h-8,fill:'#7b919a','font-size':10,'font-family':'system-ui'},'Start'));
  svg.append(element('text',{x:w-p.right,y:h-8,'text-anchor':'end',fill:'#7b919a','font-size':10,'font-family':'system-ui'},`${trace.length} recorded updates`));
}
export function exportSVG(best,area,modes) {const svg=element('svg');drawShape(svg,best,null,area,modes,true);return new XMLSerializer().serializeToString(svg);}
