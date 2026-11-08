export default function ({ parentElement, data, setStateValue }) {
  // Only original-image coordinates cross the Python boundary.
  // v2 calls the renderer again on state updates without necessarily unmounting it.
  parentElement.querySelector(".eye-editor")?.disposeEyeEditor?.();
  const root = document.createElement("section");
  root.className = "eye-editor";
  root.innerHTML = `
    <div class="eye-toolbar">
      <label>操作 <select aria-label="畫布操作"><option value="draw">描繪輪廓</option><option value="zoom">框選放大</option></select></label>
      <label>區域 <select aria-label="區域操作"><option value="add">加入眼部</option><option value="erase">扣除遮擋</option></select></label>
      <button type="button" data-action="close">完成輪廓</button>
      <button type="button" data-action="undo">復原一步</button>
      <button type="button" data-action="clear">清除輪廓</button>
      <button type="button" data-action="reset">回到動物</button>
      <label><input type="checkbox" aria-label="顯示像素格">像素格</label>
    </div>
    <p class="eye-help">先框選頭部放大，再沿眼睛邊界逐點點擊；至少三點後按「完成輪廓」。可拖曳已有頂點調整。放大不增加解析度。</p>
    <div class="eye-surface"><canvas aria-label="眼睛輪廓畫布" role="img"></canvas></div>
    <p class="eye-status" role="status"></p>`;
  parentElement.appendChild(root);
  const style = document.createElement("style");
  style.textContent = `
    .eye-editor{font:14px system-ui;color:var(--st-text-color,#253238)}
    .eye-toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:8px}
    .eye-toolbar button,.eye-toolbar select{font:inherit;padding:7px 10px;border:1px solid #bcc8ca;border-radius:6px;background:var(--st-background-color,#fff);color:inherit;cursor:pointer}
    .eye-toolbar label{display:flex;align-items:center;gap:5px}
    .eye-help,.eye-status{line-height:1.55;color:var(--st-text-color,#4a595d);margin:7px 0}
    .eye-surface{border:1px solid #bac6c9;border-radius:8px;background:#182025;overflow:auto;display:flex;justify-content:center}
    canvas{display:block;max-width:100%;touch-action:none;cursor:crosshair;image-rendering:pixelated}`;
  root.appendChild(style);
  const canvas = root.querySelector("canvas"), ctx = canvas.getContext("2d");
  const mode = root.querySelector('[aria-label="畫布操作"]');
  const operation = root.querySelector('[aria-label="區域操作"]');
  const grid = root.querySelector('[aria-label="顯示像素格"]');
  const status = root.querySelector(".eye-status");
  const geometry = structuredClone(data.geometry);
  operation.value = geometry.operation;
  grid.checked = Boolean(geometry.grid);
  let dragging = null, zoomStart = null, zoomEnd = null;
  const source = new Image();
  const emit = () => {
    const box = canvas.getBoundingClientRect(), [x0,y0,x1,y1] = geometry.view;
    geometry.display_transform = {
      roi: [...geometry.view], css_width: box.width, css_height: box.height,
      scale_x: box.width/(x1-x0), scale_y: box.height/(y1-y0), padding: [0,0],
      device_pixel_ratio: window.devicePixelRatio,
    };
    setStateValue("geometry", structuredClone(geometry));
  };
  function point(event) {
    const box = canvas.getBoundingClientRect(), [x0,y0,x1,y1] = geometry.view;
    return [Math.max(0, Math.min(data.width, x0 + (event.clientX-box.left)/box.width*(x1-x0))),
            Math.max(0, Math.min(data.height, y0 + (event.clientY-box.top)/box.height*(y1-y0)))];
  }
  function screen(p) {
    const [x0,y0,x1,y1] = geometry.view;
    return [(p[0]-x0)*canvas.width/(x1-x0),(p[1]-y0)*canvas.height/(y1-y0)];
  }
  function drawPath(points, closed, erase) {
    if (!points.length) return;
    ctx.beginPath();points.forEach((p,i)=>{const q=screen(p);i?ctx.lineTo(...q):ctx.moveTo(...q);});
    if (closed) {ctx.closePath();ctx.fillStyle=erase?"rgba(233,89,95,.25)":data.fill;ctx.fill("evenodd");}
    ctx.strokeStyle=erase?"#ff696e":data.color;ctx.lineWidth=2;ctx.stroke();
    for (const p of points) {const q=screen(p);ctx.beginPath();ctx.arc(...q,3.5,0,Math.PI*2);ctx.fillStyle="#fff";ctx.fill();ctx.stroke();}
  }
  function render() {
    if (!source.complete || !source.naturalWidth) return;
    const [x0,y0,x1,y1]=geometry.view, w=x1-x0, h=y1-y0;
    const scale=Math.min(Math.max(260,root.clientWidth-2)/w,580/h,32);
    canvas.width=Math.max(1,Math.round(w*scale));canvas.height=Math.max(1,Math.round(h*scale));
    ctx.imageSmoothingEnabled=false;ctx.drawImage(source,x0,y0,w,h,0,0,canvas.width,canvas.height);
    if (grid.checked && scale>=5) {
      ctx.lineWidth=.5;ctx.strokeStyle="rgba(255,255,255,.35)";ctx.beginPath();
      for(let x=Math.ceil(x0);x<x1;x++){const q=screen([x,y0]);ctx.moveTo(q[0],0);ctx.lineTo(q[0],canvas.height);}
      for(let y=Math.ceil(y0);y<y1;y++){const q=screen([x0,y]);ctx.moveTo(0,q[1]);ctx.lineTo(canvas.width,q[1]);}ctx.stroke();
    }
    geometry.contours.forEach(c=>drawPath(c.points,true,c.operation==="erase"));
    drawPath(geometry.active,false,geometry.operation==="erase");
    if(zoomStart && zoomEnd){const a=screen(zoomStart),b=screen(zoomEnd);ctx.strokeStyle="#fff";ctx.strokeRect(a[0],a[1],b[0]-a[0],b[1]-a[1]);}
    status.textContent=`${geometry.contours.length} 條已完成輪廓 · ${geometry.active.length} 個未完成頂點 · 原圖區域 (${x0.toFixed(1)}, ${y0.toFixed(1)})–(${x1.toFixed(1)}, ${y1.toFixed(1)})`;
  }
  canvas.onpointerdown = event => {
    canvas.setPointerCapture(event.pointerId);
    const p=point(event);
    if(mode.value==="zoom"){zoomStart=p;zoomEnd=p;return;}
    const groups=[...geometry.contours.map(c=>c.points),geometry.active];
    for(let g=0;g<groups.length;g++) for(let i=0;i<groups[g].length;i++) {
      const q=screen(groups[g][i]),r=screen(p);
      if(Math.hypot(q[0]-r[0],q[1]-r[1])<8){dragging={points:groups[g],index:i};return;}
    }
    dragging={add:true,start:p};
  };
  canvas.onpointermove = event => {
    if(zoomStart){zoomEnd=point(event);render();}
    else if(dragging && !dragging.add){dragging.points[dragging.index]=point(event);render();}
  };
  canvas.onpointerup = event => {
    if(zoomStart) {
      const end=point(event),a=zoomStart;
      const next=[Math.min(a[0],end[0]),Math.min(a[1],end[1]),Math.max(a[0],end[0]),Math.max(a[1],end[1])];
      if(next[2]-next[0]>=2 && next[3]-next[1]>=2) geometry.view=next;
      zoomStart=null;zoomEnd=null;mode.value="draw";
    } else if(dragging?.add) geometry.active.push(point(event));
    dragging=null;render();emit();
  };
  canvas.onpointercancel=()=>{dragging=null;zoomStart=null;zoomEnd=null;render();};
  operation.onchange=()=>{geometry.operation=operation.value;emit();};
  grid.onchange=()=>{geometry.grid=grid.checked;render();emit();};
  root.querySelector('[data-action="close"]').onclick=()=>{
    if(geometry.active.length<3){status.textContent="至少三個不同頂點才能完成輪廓。";return;}
    geometry.contours.push({operation:geometry.operation,points:geometry.active});geometry.active=[];render();emit();
  };
  root.querySelector('[data-action="undo"]').onclick=()=>{
    if(geometry.active.length)geometry.active.pop();else geometry.contours.pop();render();emit();
  };
  root.querySelector('[data-action="clear"]').onclick=()=>{geometry.contours=[];geometry.active=[];render();emit();};
  root.querySelector('[data-action="reset"]').onclick=()=>{geometry.view=[...data.initial_view];render();emit();};
  const observer=new ResizeObserver(render);observer.observe(root);
  source.onload=render;source.src=data.image_url;
  root.disposeEyeEditor=()=>{observer.disconnect();source.onload=null;root.remove();};
  return root.disposeEyeEditor;
}
