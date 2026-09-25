export const norm = a => Math.hypot(...a);
export const dot = (a,b) => a.reduce((s,v,i)=>s+v*b[i],0);
export function amplitudes(q) {const m=q.length/2; return Float64Array.from({length:m},(_,j)=>Math.hypot(q[j],q[m+j]));}
export function project(q, rho, caps, metric=null) {
  const m=q.length/2, a=amplitudes(q), weights=metric || new Float64Array(m).fill(1);
  let clipped=Float64Array.from(a,(v,i)=>Math.min(v,caps[i]));
  if (clipped.reduce((s,v)=>s+v,0)>rho) {
    let lo=0, hi=Math.max(...a.map((v,i)=>v/weights[i]));
    for (let step=0;step<60;step++) {
      const mid=(lo+hi)/2; let sum=0;
      for(let j=0;j<m;j++) sum+=Math.min(Math.max(a[j]-mid*weights[j],0),caps[j]);
      if (sum>rho) lo=mid; else hi=mid;
    }
    clipped=Float64Array.from(a,(v,j)=>Math.min(Math.max(v-hi*weights[j],0),caps[j]));
  }
  return Float64Array.from(q,(v,i)=>v*clipped[i%m]/Math.max(a[i%m],1e-30));
}
export function coefficients(q, area=1) {
  const a0=Math.sqrt(area/(Math.PI*(1+dot(q,q)/2)));
  return [a0,...Array.from(q,v=>a0*v)];
}
export function boundary(c, count=360) {
  const m=(c.length-1)/2, points=[];
  for (let i=0;i<count;i++) {
    const theta=2*Math.PI*i/count; let r=c[0];
    for (let j=1;j<=m;j++) r+=c[j]*Math.cos(j*theta)+c[m+j]*Math.sin(j*theta);
    points.push([r*Math.cos(theta),r*Math.sin(theta)]);
  }
  return points;
}
export function rotate(q, angle, reflect=false) {
  const m=q.length/2, out=new Float64Array(q.length);
  for (let j=0;j<m;j++) {
    const co=Math.cos((j+1)*angle), si=Math.sin((j+1)*angle);
    out[j]=q[j]*co-q[j+m]*si; out[j+m]=(q[j]*si+q[j+m]*co)*(reflect?-1:1);
  }
  return out;
}
