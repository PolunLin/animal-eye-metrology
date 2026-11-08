export default function ({ parentElement, data, setStateValue }) {
  parentElement.querySelector(".prompt-editor")?.dispose?.();
  const root = document.createElement("section");
  root.className = "prompt-editor";
  root.innerHTML = `<div class="prompt-tools">
    <label>操作 <select aria-label="提示畫布操作"><option value="zoom">框選放大</option><option value="box">畫眼部粗框</option><option value="point">點眼內正點</option></select></label>
    <button type="button" data-action="reset">回到動物</button>
    <button type="button" data-action="clear">清除本眼提示</button>
    </div><p>先框選放大頭部，再畫一個眼部粗框，最後點一個框內的眼部位置。只依原圖判斷。</p>
    <div class="prompt-surface"><canvas aria-label="原圖眼部提示畫布" role="img"></canvas></div>
    <p role="status"></p>`;
  parentElement.appendChild(root);
  const style = document.createElement("style");
  style.textContent = `.prompt-editor{font:14px system-ui;color:var(--st-text-color,#263238)}
    .prompt-tools{display:flex;flex-wrap:wrap;gap:10px;align-items:center}
    .prompt-tools button,.prompt-tools select{padding:8px;border:1px solid #bbc5ce;border-radius:6px;background:var(--st-background-color,#fff);color:inherit}
    .prompt-surface{background:#17202b;display:flex;justify-content:center;overflow:hidden;border-radius:8px}
    .prompt-surface canvas{max-width:100%;display:block;touch-action:none;cursor:crosshair;image-rendering:pixelated}`;
  root.appendChild(style);
  const canvas = root.querySelector("canvas"), ctx = canvas.getContext("2d");
  const mode = root.querySelector("select"), status = root.querySelector('[role="status"]');
  const geometry = structuredClone(data.geometry);
  let start = null, end = null;
  const source = new Image();
  function point(event) {
    const r = canvas.getBoundingClientRect(), [x0,y0,x1,y1] = geometry.view;
    return [Math.max(0,Math.min(data.width,x0+(event.clientX-r.left)/r.width*(x1-x0))),
      Math.max(0,Math.min(data.height,y0+(event.clientY-r.top)/r.height*(y1-y0)))];
  }
  function screen(p) {
    const [x0,y0,x1,y1] = geometry.view;
    return [(p[0]-x0)/(x1-x0)*canvas.width,(p[1]-y0)/(y1-y0)*canvas.height];
  }
  function rectangle(box,color) {
    const a=screen(box.slice(0,2)),b=screen(box.slice(2));
    ctx.strokeStyle=color;ctx.lineWidth=2;ctx.strokeRect(a[0],a[1],b[0]-a[0],b[1]-a[1]);
  }
  function render() {
    if(!source.complete || !source.naturalWidth) return;
    const [x0,y0,x1,y1]=geometry.view,w=x1-x0,h=y1-y0;
    const scale=Math.min(Math.max(260,root.clientWidth-2)/w,580/h,32);
    canvas.width=Math.max(1,Math.round(w*scale));canvas.height=Math.max(1,Math.round(h*scale));
    ctx.imageSmoothingEnabled=false;ctx.drawImage(source,x0,y0,w,h,0,0,canvas.width,canvas.height);
    if(geometry.box)rectangle(geometry.box,"#00b9c9");
    if(geometry.point){const p=screen(geometry.point);ctx.beginPath();ctx.arc(p[0],p[1],4,0,2*Math.PI);ctx.fillStyle="#ffb000";ctx.fill();ctx.strokeStyle="black";ctx.stroke();}
    if(start&&end)rectangle([Math.min(start[0],end[0]),Math.min(start[1],end[1]),Math.max(start[0],end[0]),Math.max(start[1],end[1])],"white");
    const fmt=p=>p?p.map(v=>v.toFixed(2)).join(", "):"尚未設定";
    status.textContent=`粗框：${fmt(geometry.box)} · 正點：${fmt(geometry.point)} · 座標為原圖 px`;
  }
  const emit=()=>setStateValue("geometry",structuredClone(geometry));
  canvas.onpointerdown=e=>{canvas.setPointerCapture(e.pointerId);start=point(e);end=start;};
  canvas.onpointermove=e=>{if(start){end=point(e);render();}};
  canvas.onpointerup=e=>{
    if(!start)return;
    end=point(e);
    if(mode.value==="point")geometry.point=end;
    else {
      const box=[Math.min(start[0],end[0]),Math.min(start[1],end[1]),Math.max(start[0],end[0]),Math.max(start[1],end[1])];
      if(box[2]-box[0]>=1&&box[3]-box[1]>=1){
        if(mode.value==="zoom"){geometry.view=box;mode.value="box";}else{geometry.box=box;mode.value="point";}
      }
    }
    start=null;end=null;render();emit();
  };
  canvas.onpointercancel=()=>{start=null;end=null;render();};
  root.querySelector('[data-action="reset"]').onclick=()=>{geometry.view=[...data.initial_view];mode.value="zoom";render();emit();};
  root.querySelector('[data-action="clear"]').onclick=()=>{geometry.box=null;geometry.point=null;render();emit();};
  const observer=new ResizeObserver(render);observer.observe(root);
  source.onload=render;source.src=data.image_url;
  root.dispose=()=>{observer.disconnect();source.onload=null;root.remove();};
  return root.dispose;
}
