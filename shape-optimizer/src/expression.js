/** Restricted expression grammar and symbolic differentiation. Never executes input as JS. */
const FUNCTIONS = new Set(['sin','cos','tan','asin','acos','atan','sinh','cosh','tanh','exp','log','sqrt']);
const number = value => ({type:'number', value});
const zero = node => node.type === 'number' && node.value === 0;
const one = node => node.type === 'number' && node.value === 1;
function op(type, a, b) {
  // Simplification is used only for derivative trees; original domain checks remain intact.
  if (type === '+' && zero(a)) return b;
  if ((type === '+' || type === '-') && zero(b)) return a;
  if (type === '*' && (zero(a) || zero(b))) return number(0);
  if ((type === '*' || type === '/') && one(b)) return a;
  if (type === '*' && one(a)) return b;
  if (type === '/' && zero(a)) return number(0);
  if (type === '^' && zero(b)) return number(1);
  if (type === '^' && one(b)) return a;
  return {type, a, b};
}
const call = (name, a) => ({type:'call', name, a});
const neg = a => op('*', number(-1), a);

export function parseExpression(source) {
  if (typeof source !== 'string' || !source.trim() || source.length > 1024) throw Error('Enter a formula of 1–1024 characters.');
  const tokens = []; let pos = 0;
  while (pos < source.length) {
    if (/\s/.test(source[pos])) {pos++; continue;}
    const match = /^(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|^[A-Za-z][A-Za-z0-9]*|^\*\*|^[+\-*/^()]/.exec(source.slice(pos));
    if (!match) throw Error(`Unexpected character at position ${pos + 1}. Use x1…x10, arithmetic and the listed functions.`);
    tokens.push(match[0] === '**' ? '^' : match[0]); pos += match[0].length;
    if (tokens.length > 256) throw Error('Formula is too complex (maximum 256 tokens).');
  }
  let i = 0, depth = 0;
  const raw = (type,a,b) => {
    if(a.type==='number'&&b.type==='number') {
      const v=type==='+'?a.value+b.value:type==='-'?a.value-b.value:type==='*'?a.value*b.value:type==='/'?a.value/b.value:a.value**b.value;
      if(Number.isFinite(v))return number(v);
    }
    return {type,a,b};
  };
  function primary() {
    if (++depth > 32) throw Error('Formula nesting exceeds 32 levels.');
    const t = tokens[i++]; let result;
    if (t === '(') {result = sum(); if (tokens[i++] !== ')') throw Error('Missing closing parenthesis.');}
    else if (t && /^(?:\d|\.)/.test(t)) {const value = Number(t); if (!Number.isFinite(value)) throw Error('Numeric constants must be finite.'); result = number(value);}
    else if (t && /^x(?:[1-9]|10)$/.test(t)) result = {type:'variable', index:Number(t.slice(1))-1};
    else if (t === 'pi' || t === 'e') result = number(t === 'pi' ? Math.PI : Math.E);
    else if (FUNCTIONS.has(t) || t === 'ln') {
      if (tokens[i++] !== '(') throw Error(`${t} requires parentheses.`);
      const name=t === 'ln' ? 'log' : t, argument=sum();
      const constant=argument.type==='number'?Math[name](argument.value):NaN;
      result = Number.isFinite(constant)?number(constant):call(name,argument);
      if (tokens[i++] !== ')') throw Error('Functions take exactly one parenthesized argument.');
    } else throw Error(`Unknown term ${t ?? '(end of formula)'}. Variables are x1 through x10.`);
    depth--; return result;
  }
  function power() {let a = primary(); if (tokens[i] === '^') {i++; a = raw('^',a,unary());} return a;}
  function unary() {
    if (tokens[i] === '+' || tokens[i] === '-') {
      if (++depth > 32) throw Error('Formula nesting exceeds 32 levels.');
      const t=tokens[i++], a=unary(); depth--; return t === '-' ? raw('*',number(-1),a) : a;
    }
    return power();
  }
  function product() {let a=unary(); while (tokens[i] === '*' || tokens[i] === '/') {const t=tokens[i++]; a=raw(t,a,unary());} return a;}
  function sum() {let a=product(); while (tokens[i] === '+' || tokens[i] === '-') {const t=tokens[i++]; a=raw(t,a,product());} return a;}
  const root=sum(); if (i !== tokens.length) throw Error('Unexpected term. Multiplication must be explicit, for example 2*x5.');
  return root;
}

export function differentiate(n, k) {
  if (n.type === 'number') return number(0);
  if (n.type === 'variable') return number(n.index === k ? 1 : 0);
  const a=n.a, da=differentiate(a,k), b=n.b;
  if (n.type === 'call') {
    if (zero(da)) return number(0);
    let outer;
    switch (n.name) {
      case 'sin': outer=call('cos',a); break;
      case 'cos': outer=neg(call('sin',a)); break;
      case 'tan': outer=op('/',number(1),op('^',call('cos',a),number(2))); break;
      case 'asin': outer=op('/',number(1),call('sqrt',op('-',number(1),op('^',a,number(2))))); break;
      case 'acos': outer=neg(op('/',number(1),call('sqrt',op('-',number(1),op('^',a,number(2)))))); break;
      case 'atan': outer=op('/',number(1),op('+',number(1),op('^',a,number(2)))); break;
      case 'sinh': outer=call('cosh',a); break;
      case 'cosh': outer=call('sinh',a); break;
      case 'tanh': outer=op('-',number(1),op('^',call('tanh',a),number(2))); break;
      case 'exp': outer=call('exp',a); break;
      case 'log': outer=op('/',number(1),a); break;
      case 'sqrt': outer=op('/',number(1),op('*',number(2),call('sqrt',a))); break;
    }
    return op('*',outer,da);
  }
  const db=differentiate(b,k);
  if (n.type === '+' || n.type === '-') return op(n.type,da,db);
  if (n.type === '*') return op('+',op('*',da,b),op('*',a,db));
  if (n.type === '/') return op('/',op('-',op('*',da,b),op('*',a,db)),op('^',b,number(2)));
  if (n.type === '^') {
    if (zero(da) && zero(db)) return number(0);
    if (b.type === 'number') return op('*',op('*',b,op('^',a,number(b.value-1))),da);
    return op('*',n,op('+',op('*',db,call('log',a)),op('*',b,op('/',da,a))));
  }
  throw Error('Unsupported expression node.');
}

export class DomainError extends Error {}
export function evaluateTree(n, x) {
  let v;
  if (n.type === 'number') v=n.value;
  else if (n.type === 'variable') v=x[n.index];
  else if (n.type === 'call') v=Math[n.name](evaluateTree(n.a,x));
  else {
    const a=evaluateTree(n.a,x), b=evaluateTree(n.b,x);
    switch(n.type) {case '+':v=a+b;break;case '-':v=a-b;break;case '*':v=a*b;break;case '/':v=a/b;break;case '^':v=a**b;break;}
  }
  if (!Number.isFinite(v)) throw new DomainError('F or its derivative is undefined at this spectrum. Check denominators, logarithms, roots and powers.');
  return v;
}
export function printTree(n) {
  if (n.type === 'number') return String(n.value);
  if (n.type === 'variable') return `x${n.index+1}`;
  if (n.type === 'call') return `${n.name}(${printTree(n.a)})`;
  return `(${printTree(n.a)} ${n.type} ${printTree(n.b)})`;
}
export function compileExpression(source) {
  const tree=parseExpression(source), derivatives=Array.from({length:10},(_,k)=>differentiate(tree,k));
  return {source, tree, derivatives:derivatives.map(printTree), evaluate(x) {
    if (x.length !== 10 || !Array.from(x).every(v=>Number.isFinite(v) && v>0)) throw new DomainError('The objective requires ten positive finite eigenvalues.');
    return {value:evaluateTree(tree,x), partials:Float64Array.from(derivatives,d=>evaluateTree(d,x))};
  }};
}
