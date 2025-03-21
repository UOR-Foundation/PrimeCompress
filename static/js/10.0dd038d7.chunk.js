"use strict";(self.webpackChunkprimecompress_frontend=self.webpackChunkprimecompress_frontend||[]).push([[10],{10:(t,e,r)=>{r.r(e),r.d(e,{WasmStatus:()=>n,default:()=>a});let n=function(t){return t.NOT_LOADED="not_loaded",t.LOADING="loading",t.LOADED="loaded",t.ERROR="error",t}({});class s{constructor(){this.status=n.NOT_LOADED,this.error=null,this.wasmModule=null,this.loadPromise=null}static getInstance(){return s.instance||(s.instance=new s),s.instance}getStatus(){return this.status}getError(){return this.error}async load(){return this.loadPromise?this.loadPromise:this.status===n.LOADED?Promise.resolve():(this.status=n.LOADING,this.loadPromise=new Promise(((t,e)=>{console.log("Loading PrimeCompress WebAssembly module...");try{const e={pattern:this.patternCompress.bind(this),sequential:this.sequentialCompress.bind(this),spectral:this.spectralCompress.bind(this),dictionary:this.dictionaryCompress.bind(this),auto:this.autoCompress.bind(this)};this.wasmModule={compress:this.realCompress.bind(this,e),decompress:this.realDecompress.bind(this),getAvailableStrategies:this.realGetAvailableStrategies.bind(this)},console.log("PrimeCompress module loaded successfully"),this.status=n.LOADED,t()}catch(r){console.error("Failed to load WebAssembly module:",r),this.status=n.ERROR,this.error=r instanceof Error?r:new Error(String(r)),e(this.error)}})),this.loadPromise)}async compress(t){let e=arguments.length>1&&void 0!==arguments[1]?arguments[1]:{};return this.status!==n.LOADED&&await this.load(),this.wasmModule.compress(t,e)}async decompress(t){return this.status!==n.LOADED&&await this.load(),this.wasmModule.decompress(t)}async getAvailableStrategies(){return this.status!==n.LOADED&&await this.load(),this.wasmModule.getAvailableStrategies()}calculateChecksum(t){let e=0;for(let r=0;r<t.length;r++){e=(e<<5)-e+t[r],e|=0}return(e>>>0).toString(16).padStart(8,"0")}calculateEntropy(t){const e=new Array(256).fill(0);for(let n=0;n<t.length;n++)e[t[n]]++;let r=0;for(let n=0;n<256;n++)if(e[n]>0){const s=e[n]/t.length;r-=s*Math.log2(s)}return r}autoCompress(t){if(t.length>5242880)return{strategy:"dictionary",entropyScore:7};const e=this.calculateEntropy(t),r=this.analyzeBlock(t),n={pattern:0,sequential:0,dictionary:0,spectral:0};r.isConstant&&(n.pattern+=100),r.hasPattern&&(n.pattern+=50),r.hasSequence&&(n.sequential+=70),r.isTextLike&&(n.dictionary+=60),e<3?n.pattern+=40:e<5?(n.sequential+=30,n.dictionary+=20):e<7?n.dictionary+=40:n.spectral+=50,t.length<1024?n.pattern+=15:t.length>102400&&(n.dictionary+=20),r.hasSpectralPattern&&(n.spectral+=40);let s="dictionary",a=n.dictionary;return n.pattern>a&&(s="pattern",a=n.pattern),n.sequential>a&&(s="sequential",a=n.sequential),n.spectral>a&&(s="spectral",a=n.spectral),e>7&&a<30&&(s="spectral"),console.debug(`Selected strategy: ${s} (entropy: ${e.toFixed(2)}, scores: `,n,`, isConstant: ${r.isConstant}, hasPattern: ${r.hasPattern}, hasSequence: ${r.hasSequence}, isTextLike: ${r.isTextLike})`),{strategy:s,entropyScore:e}}analyzeBlock(t){const e={entropy:this.calculateEntropy(t),isConstant:!0,hasPattern:!1,hasSequence:!1,hasSpectralPattern:!1,isTextLike:!1};if(t.length>0){const r=t[0];for(let n=1;n<t.length;n++)if(t[n]!==r){e.isConstant=!1;break}}return!e.isConstant&&t.length>=8&&(e.hasPattern=this.detectPattern(t)),!e.isConstant&&!e.hasPattern&&t.length>=8&&(e.hasSequence=this.detectSequence(t)),!e.isConstant&&t.length>=16&&(e.hasSpectralPattern=this.detectSpectralPattern(t)),t.length>0&&(e.isTextLike=this.isTextLike(t)),e}detectPattern(t){if(t.length<8)return!1;const e=t.length>4096?Math.ceil(t.length/4096):1,r=t[0];let n=!0;const s=Math.min(64,t.length);for(let p=1;p<s;p++){if(t[p*e<t.length?p*e:p]!==r){n=!1;break}}if(n&&s>=16)return!0;let a=Math.max(8,Math.floor(.15*t.length)),o=t[0],l=1,i=1,c=0;const h=Math.min(t.length,1e4);for(let p=1;p<h;p++)t[p]===o?l++:(l>=4&&(c+=l),o=t[p],l=1),l>i&&(i=l);const u=c/Math.min(t.length,h);if(i>=a||u>.2)return!0;const g=Math.min(t.length,1024);for(let p=2;p<=8;p++){if(g%p!==0&&p>4)continue;let e=!0,r=0;const n=Math.min(8,Math.floor(g/p));for(let s=p;s<g;s++){if(t[s]!==t[s%p]){e=!1;break}if(s%p===0&&(r++,r>=n))break}if(e&&r>=n)return!0}return!1}detectSequence(t){if(t.length<8)return!1;const e=Math.min(256,t.length),r=t.length>1e3?Math.floor(t.length/100):1,n=[];if(r>1)for(let a=r;a<e;a+=r)n.push((t[a]-t[a-r]+256)%256);else for(let a=1;a<e;a++)n.push((t[a]-t[a-1]+256)%256);if(0===n.length)return!1;const s=n[0];if(n.length>32){return n.filter((t=>t===s)).length/n.length>.9}return n.every((t=>t===s))}detectSpectralPattern(t){if(t.length<16)return!1;const e=[];for(let s=1;s<t.length;s++)e.push((t[s]-t[s-1]+256)%256);let r=0,n=e[0]>0?1:e[0]<0?-1:0;for(let s=1;s<e.length;s++){const t=e[s],a=t>0?1:t<0?-1:0;0!==a&&0!==n&&a!==n&&r++,0!==a&&(n=a)}return r>=4}isTextLike(t){if(0===t.length)return!1;let e=0,r=0,n=0;const s=Math.min(t.length,100);for(let a=0;a<s;a++)t[a]>=32&&t[a]<=126&&(e++,t[a]>=65&&t[a]<=90||t[a]>=97&&t[a]<=122?r++:32===t[a]&&n++);return e/s>.7&&(r+n)/s>.4}patternCompress(t){if(0===t.length)return new Uint8Array(0);let e=!0;const r=t[0];for(let l=1;l<t.length;l++)if(t[l]!==r){e=!1;break}if(e){const e=new Uint8Array(3);return e[0]=192,e[1]=r,e[2]=Math.min(255,t.length),e}let n=!1,s=[];for(let l=2;l<=16;l++){if(t.length<2*l)continue;let e=!0;s=Array.from(t.slice(0,l));for(let r=l;r<t.length;r++)if(t[r]!==s[r%l]){e=!1;break}if(e){n=!0;break}}if(n&&s.length>0&&s.length<=16){const e=new Uint8Array(3+s.length);e[0]=193,e[1]=s.length,e[2]=Math.floor(t.length/s.length);for(let t=0;t<s.length;t++)e[3+t]=s[t];return e}const a=[];let o=0;for(a.push(240);o<t.length;){let e=1;const r=t[o];for(;o+e<t.length&&t[o+e]===r&&e<255;)e++;if(e>=4)a.push(255),a.push(e),a.push(r),o+=e;else{let e=1,r=Math.min(127,t.length-o);for(;e<r;){const r=t[o+e];let n=1;for(;o+e+n<t.length&&t[o+e+n]===r&&n<255;)n++;if(n>=4)break;e++}a.push(e-1);for(let n=0;n<e;n++)a.push(t[o+n]);o+=e}}return new Uint8Array(a)}sequentialCompress(t){if(t.length<4)return t;const e=[];for(let s=1;s<Math.min(16,t.length);s++)e.push((t[s]-t[s-1]+256)%256);const r=e[0];if(e.every((t=>t===r))){const e=new Uint8Array(5);return e[0]=241,e[1]=t[0],e[2]=r,e[3]=255&t.length,e[4]=t.length>>8&255,e}let n=!0;for(let s=0;s<Math.min(256,t.length);s++)if(t[s]!==s%256){n=!1;break}if(n){const e=new Uint8Array(3);return e[0]=242,e[1]=255&t.length,e[2]=t.length>>8&255,e}return this.patternCompress(t)}spectralCompress(t){if(this.calculateEntropy(t)>7){const e=new Uint8Array(t.length+3);return e[0]=243,e[1]=255&t.length,e[2]=t.length>>8&255,e.set(t,3),e}let e=255,r=0;for(let s=0;s<t.length;s++)t[s]<e&&(e=t[s]),t[s]>r&&(r=t[s]);if(r-e<=32){const n=[];n.push(244),n.push(e),n.push(r);const s=Math.ceil(Math.log2(r-e+1));if(Math.floor(8/s)>=2){n.push(s);let r=0,a=0;for(let o=0;o<t.length;o++){const l=t[o]-e;r|=l<<a,a+=s,a>=8&&(n.push(255&r),r=l>>s-(a-8),a-=8)}return a>0&&n.push(255&r),new Uint8Array(n)}}const n=[];n.push(245),n.push(t[0]);for(let s=1;s<t.length;s++){let e=t[s]-t[s-1];e<-128&&(e+=256),e>127&&(e-=256),n.push(255&e)}return new Uint8Array(n)}dictionaryCompress(t){if(t.length<8)return t;const e=1e5,r=Math.min(t.length,e);let n=t,s=1;if(t.length>e){s=Math.ceil(t.length/e);const r=Math.min(e,Math.ceil(t.length/s)+1e3);n=new Uint8Array(r);const a=Math.min(1e3,t.length);for(let e=0;e<a;e++)n[e]=t[e];let o=a;for(let e=1e3;e<t.length-1e3;e+=s)o<n.length&&(n[o++]=t[e]);const l=Math.max(0,t.length-1e3);let i=n.length-o;for(let e=0;e<i&&l+e<t.length;e++)n[o++]=t[l+e]}const a=new Array(256).fill(0);for(let g=0;g<r;g++)a[t[g]]++;const o=new Map;for(let g=0;g<n.length-1;g++){const t=n[g]<<8|n[g+1];o.set(t,(o.get(t)||0)+1)}const l=t.length>1e5?24:16,i=Array.from(o.entries()).sort(((t,e)=>e[1]-t[1])).slice(0,l).map((t=>t[0]));if(0===i.length)return this.patternCompress(t);const c=i.map((t=>[t>>8,255&t]));if(t.length>1e4){const e=[];let r=0;const n=Math.min(1e4,t.length);for(;r<n;){if(r<n-1){const n=t[r]<<8|t[r+1],s=i.indexOf(n);if(s>=0){e.push(224|s),r+=2;continue}}e.push(t[r]),r++}const s=2+2*c.length,a=n/(e.length+s);if(a<1.05){if(a<1.01){const e=new Uint8Array(t.length+3);return e[0]=247,e[1]=255&t.length,e[2]=t.length>>8&255,e.set(t,3),e}return this.patternCompress(t)}}const h=[];h.push(246),h.push(c.length);for(const[g,p]of c)h.push(g),h.push(p);let u=0;for(;u<t.length;){if(u<t.length-1){const e=t[u]<<8|t[u+1],r=i.indexOf(e);if(r>=0){h.push(224|r),u+=2;continue}}h.push(t[u]),u++}if(h.length>=t.length){const e=new Uint8Array(t.length+3);return e[0]=247,e[1]=255&t.length,e[2]=t.length>>8&255,e.set(t,3),e}return new Uint8Array(h)}realCompress(t,e){let r=arguments.length>2&&void 0!==arguments[2]?arguments[2]:{};return new Promise((n=>{const s=performance.now();try{let a=r.strategy||"auto",o="";if("auto"===a){o=this.autoCompress(e).strategy,a=o}let l;if(!1!==r.useBlocks&&e.length>8192)l=this.compressWithBlocks(e,a,t);else switch(a){case"pattern":l=t.pattern(e);break;case"sequential":l=t.sequential(e);break;case"spectral":l=t.spectral(e);break;case"dictionary":l=t.dictionary(e);break;default:switch(t.auto(e).strategy){case"pattern":l=t.pattern(e);break;case"sequential":l=t.sequential(e);break;case"spectral":l=t.spectral(e);break;case"dictionary":l=t.dictionary(e);break;default:l=t.dictionary(e),a="dictionary"}}const i=e.length/l.length,c=performance.now()-s;n({compressedData:l,compressionRatio:i,strategy:a,originalSize:e.length,compressedSize:l.length,compressionTime:c})}catch(a){console.error("Compression error:",a);const t=a instanceof Error?a.message:String(a);throw new Error(`Compression failed: ${t||"Unknown error"}`)}}))}compressWithBlocks(t,e,r){let n;n=t.length>104857600?1048576:t.length>10485760?262144:t.length>1048576?65536:t.length>102400?32768:16384;const s=Math.ceil(t.length/n);s>1e3&&(n=Math.ceil(t.length/1e3),console.debug(`Adjusted block size to ${n} bytes to limit block count`));const a=[],o=[],l=[],i=new Uint8Array(7);i[0]=177,i[1]=255&s,i[2]=s>>8&255,i[3]=255&n,i[4]=n>>8&255,i[5]=n>>16&255,i[6]=n>>24&255,a.push(i);const c={pattern:0,sequential:0,spectral:0,dictionary:0};for(let y=0;y<s;y++){const i=y*n,h=Math.min(i+n,t.length),u=t.slice(i,h),g=t.length<52428800||y%5===0||y<10||y>=s-5;let p,d=e;"auto"===e&&g?d=this.autoCompress(u).strategy:"auto"===e&&(d="dictionary");try{switch(d){case"pattern":p=r.pattern(u);break;case"sequential":p=r.sequential(u);break;case"spectral":p=r.spectral(u);break;case"dictionary":p=r.dictionary(u);break;default:p=r.dictionary(u),d="dictionary"}}catch(m){console.error(`Error compressing block ${y}, falling back to dictionary:`,m),p=r.dictionary(u),d="dictionary"}c[d]++,p.length>=u.length&&(p=new Uint8Array(u.length+3),p[0]=247,p[1]=255&u.length,p[2]=u.length>>8&255,p.set(u,3),d="raw",c.raw||(c.raw=0),c.raw++);const f=new Uint8Array(5),w=this.getStrategyId(d);f[0]=w,f[1]=255&p.length,f[2]=p.length>>8&255,f[3]=p.length>>16&255,f[4]=p.length>>24&255,a.push(f),a.push(p),o.push(d),l.push(p.length),s>20&&(0===y||y===s-1||y%10===0)&&console.debug(`Block ${y+1}/${s} compressed`)}console.debug("Block compression strategy usage:",c);let h=7;for(let y=0;y<s;y++)h+=5,h+=l[y];const u=new Uint8Array(h);let g=0;for(const y of a)u.set(y,g),g+=y.length;const p=t.length,d=u.length,f=p/d;return console.debug(`Block compression complete: ${p} \u2192 ${d} bytes (${f.toFixed(2)}x ratio)`),u}getStrategyId(t){switch(t){case"pattern":return 1;case"sequential":return 2;case"spectral":return 3;case"dictionary":return 4;case"raw":return 5;default:return 0}}realDecompress(t){return new Promise(((e,r)=>{try{if(0===t.length)return e(new Uint8Array(0));const n=t[0];if(177===n)return this.decompressBlocks(t).then(e).catch(r);switch(n){case 192:if(t.length<3)return r(new Error("Invalid constant data format"));const n=t[1],s=t[2],a=new Uint8Array(s);return a.fill(n),e(a);case 193:if(t.length<3)return r(new Error("Invalid pattern data format"));const o=t[1],l=t[2];if(t.length<3+o)return r(new Error("Invalid pattern data length"));const i=t.slice(3,3+o),c=new Uint8Array(o*l);for(let t=0;t<l;t++)c.set(i,t*o);return e(c);case 240:{const r=[];let n=1;for(;n<t.length;)if(255===t[n]&&n+2<t.length){const e=t[n+1],s=t[n+2];for(let t=0;t<e;t++)r.push(s);n+=3}else if(t[n]<128){const e=t[n]+1;for(let s=0;s<e&&n+1+s<t.length;s++)r.push(t[n+1+s]);n+=e+1}else r.push(t[n]),n++;return e(new Uint8Array(r))}case 241:if(t.length<5)return r(new Error("Invalid arithmetic sequence data"));const h=t[1],u=t[2],g=t[3],p=g|t[4]<<8,d=new Uint8Array(p);let f=h;for(let t=0;t<p;t++)d[t]=f,f=(f+u)%256;return e(d);case 242:if(t.length<3)return r(new Error("Invalid modulo sequence data"));const m=t[1],y=m|t[2]<<8,w=new Uint8Array(y);for(let t=0;t<y;t++)w[t]=t%256;return e(w);case 243:if(t.length<3)return r(new Error("Invalid high-entropy data format"));const b=t[1],k=b|t[2]<<8;return t.length<3+k?r(new Error("Invalid high-entropy data length")):e(t.slice(3,3+k));case 244:if(t.length<4)return r(new Error("Invalid range-compressed data format"));const A=t[1],E=t[3],M=Math.floor(8/E),S=t.length-4,v=S*M,C=new Uint8Array(v);let $=0;const q=(1<<E)-1;for(let e=0;e<S&&$<v;e++){const r=t[e+4];for(let t=0;t<M&&$<v;t++){const e=r>>t*E&q;C[$++]=A+e}}return e(C);case 245:if(t.length<2)return r(new Error("Invalid delta-encoded data format"));const U=t[1],D=new Uint8Array(t.length-1);D[0]=U;for(let e=1;e<D.length;e++){const r=t[e+1],n=r<=127?r:r-256;D[e]=D[e-1]+n&255}return e(D);case 246:if(t.length<3)return r(new Error("Invalid dictionary compressed data"));const I=t[1];if(0===I||t.length<2+2*I)return r(new Error("Invalid dictionary size"));const P=[];let L=2;for(let e=0;e<I;e++)P.push([t[L],t[L+1]]),L+=2;const x=[];for(;L<t.length;)if(224===(240&t[L])){const e=15&t[L];e<P.length?x.push(P[e][0],P[e][1]):x.push(t[L]),L++}else x.push(t[L]),L++;return e(new Uint8Array(x));case 247:if(t.length<3)return r(new Error("Invalid uncompressed data format"));const O=t[1],B=O|t[2]<<8;return t.length<3+B?r(new Error("Invalid uncompressed data length")):e(t.slice(3,3+B));default:return e(t)}}catch(n){console.error("Decompression error:",n);const t=n instanceof Error?n.message:String(n);r(new Error(`Decompression error: ${t||"Unknown error"}`))}}))}async decompressBlocks(t){if(t.length<7)throw new Error("Invalid block-compressed data format: Header too small");if(177!==t[0])throw new Error("Invalid block-compressed data format: Invalid marker byte");const e=t[1]|t[2]<<8,r=t[3]|t[4]<<8|t[5]<<16|t[6]<<24;if(console.debug(`Decompressing ${e} blocks with nominal block size of ${r} bytes`),e>1e4)throw new Error(`Invalid block count: ${e} (maximum 10000)`);const n=[];let s=7;for(let c=0;c<e;c++){if(s+5>t.length)throw new Error(`Invalid block header at block ${c}: Not enough data remaining`);const r=t[s],a=t[s+1]|t[s+2]<<8|t[s+3]<<16|t[s+4]<<24;if(s+=5,a>10485760)throw new Error(`Invalid block length at block ${c}: ${a} bytes (maximum 10MB)`);if(s+a>t.length)throw new Error(`Invalid block data at block ${c}: Expected ${a} bytes but only ${t.length-s} remaining`);const o=t.slice(s,s+a);s+=a,e>20&&(0===c||c===e-1||c%10===0)&&console.debug(`Decompressing block ${c+1}/${e}`);try{const t=this.getStrategyFromId(r);if("raw"===t||255===r){const t=await this.realDecompress(o);n.push(t);continue}if(o.length>0&&this.isValidMarker(o[0])){const t=await this.realDecompress(o);n.push(t)}else{const e=this.getMarkerForStrategy(t,o),r=new Uint8Array(o.length+1);r[0]=e,r.set(o,1);const s=await this.realDecompress(r);n.push(s)}}catch(i){throw console.error(`Error decompressing block ${c}:`,i),new Error(`Failed to decompress block ${c}: ${i instanceof Error?i.message:String(i)}`)}}let a=0;for(const c of n)a+=c.length;const o=new Uint8Array(a);let l=0;for(const c of n)o.set(c,l),l+=c.length;return console.debug(`Block decompression complete: ${t.length} \u2192 ${o.length} bytes`),o}isValidMarker(t){return[192,193,240,241,242,243,244,245,246,247,177].includes(t)}getStrategyFromId(t){switch(t){case 1:return"pattern";case 2:return"sequential";case 3:return"spectral";case 4:return"dictionary";case 5:case 255:return"raw";default:return"auto"}}getMarkerForStrategy(t,e){switch(t){case"pattern":return e.length>0&&255===e[0]?240:192;case"sequential":return 241;case"spectral":return 245;case"dictionary":return 246;default:return 247}}realGetAvailableStrategies(){return Promise.resolve([{id:"auto",name:"Auto (Best)"},{id:"pattern",name:"Pattern Recognition"},{id:"sequential",name:"Sequential"},{id:"spectral",name:"Spectral"},{id:"dictionary",name:"Dictionary"}])}}s.instance=void 0;const a=s.getInstance()}}]);
//# sourceMappingURL=10.0dd038d7.chunk.js.map