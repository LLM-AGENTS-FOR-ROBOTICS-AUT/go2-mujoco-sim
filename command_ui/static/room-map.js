/* Read-only top-down map; API row zero is the lowest map y coordinate. */
(() => {
  const canvas = document.getElementById('roomMap'), ctx = canvas.getContext('2d');
  const note = document.getElementById('mapNote'), names = document.getElementById('mapPlaces');
  function draw(view) {
    const g = view.grid, data = Uint8Array.from(atob(g.data), c => c.charCodeAt(0));
    if (data.length !== g.width * g.height) throw new Error('Incomplete map');
    const layer = document.createElement('canvas'); layer.width = g.width; layer.height = g.height;
    const pixels = layer.getContext('2d').createImageData(g.width, g.height);
    const palette = [[18,37,53], [63,89,105], [205,222,229]];
    data.forEach((v, i) => {
      const row = Math.floor(i / g.width), col = i % g.width;
      const p = ((g.height - 1 - row) * g.width + col) * 4;
      pixels.data.set([...(palette[v] || palette[0]), 255], p);
    });
    layer.getContext('2d').putImageData(pixels, 0, 0);
    const margin = 24, extent = Math.max(g.width, g.height) * g.resolution_m;
    const scale = (canvas.width - 2 * margin) / extent;
    const left = (canvas.width - g.width * g.resolution_m * scale) / 2;
    const top = (canvas.height - g.height * g.resolution_m * scale) / 2;
    const xy = p => [left + (p.x - g.origin_m[0]) * scale,
                     top + (g.height * g.resolution_m - (p.y - g.origin_m[1])) * scale];
    ctx.fillStyle = '#122535'; ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(layer, left, top, g.width * g.resolution_m * scale, g.height * g.resolution_m * scale);
    if (view.route.length) {
      ctx.beginPath(); ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = 3;
      const route = view.robot ? [[view.robot.x, view.robot.y], ...view.route] : view.route;
      route.forEach((p, i) => { const [x,y] = xy({x:p[0], y:p[1]}); i ? ctx.lineTo(x,y) : ctx.moveTo(x,y); });
      ctx.stroke();
    }
    ctx.font = '14px system-ui'; ctx.textBaseline = 'bottom';
    const label = (text,x,y) => {
      const width = ctx.measureText(text).width;
      x = Math.min(Math.max(4,x), canvas.width-width-4); y = Math.max(18,y);
      ctx.fillStyle = '#0b1b2bdd'; ctx.fillRect(x-2,y-16,width+4,19);
      ctx.fillStyle = '#fff'; ctx.fillText(text,x,y);
    };
    // Room names sit underneath everything else, in the middle of each room.
    ctx.textAlign = 'center'; ctx.fillStyle = '#9fb3c8';
    (view.rooms || []).forEach(p => { const [x,y] = xy(p); ctx.fillText(p.name.replace(/_/g,' '),x,y+7); });
    ctx.textAlign = 'start';
    view.objects.forEach(p => {
      const [x,y] = xy(p); ctx.strokeStyle = '#c084fc'; ctx.fillStyle = '#c084fc'; ctx.lineWidth = 2;
      ctx.beginPath();
      if (p.state === 'missing') { ctx.moveTo(x-4,y-4);ctx.lineTo(x+4,y+4);ctx.moveTo(x+4,y-4);ctx.lineTo(x-4,y+4);ctx.stroke(); }
      else { ctx.arc(x,y,4,0,2*Math.PI);ctx.fill(); }
      label([p.color,p.label,p.state === 'missing' ? '(missing)' : ''].filter(Boolean).join(' '),x+7,y-5);
    });
    view.places.forEach(p => {
      const [x,y] = xy(p); ctx.fillStyle = '#fbbf24'; ctx.beginPath();
      ctx.moveTo(x,y-6);ctx.lineTo(x+6,y);ctx.lineTo(x,y+6);ctx.lineTo(x-6,y);ctx.closePath();ctx.fill();
      label(p.name,x+9,y-6);
    });
    if (view.robot) {
      const [x,y] = xy(view.robot); ctx.save(); ctx.translate(x,y); ctx.rotate(-view.robot.yaw);
      ctx.beginPath(); ctx.moveTo(12,0);ctx.lineTo(-8,-7);ctx.lineTo(-5,0);ctx.lineTo(-8,7);ctx.closePath();
      ctx.fillStyle = '#2dd4bf';ctx.fill();ctx.strokeStyle='#061c23';ctx.lineWidth=2;ctx.stroke();ctx.restore();
    }
    // A scale in metres remains readable as the observed map grows.
    const metres = extent > 10 ? 2 : 1; ctx.strokeStyle='#d5e6ef';ctx.lineWidth=2;
    ctx.beginPath();ctx.moveTo(16,canvas.height-18);ctx.lineTo(16+metres*scale,canvas.height-18);ctx.stroke();
    ctx.fillStyle='#d5e6ef';ctx.fillText(metres+' m',16,canvas.height-24);
    const loc = view.robot && view.robot.localization;
    note.textContent = view.error || (loc ? 'Scan correction: '+loc.state : 'Live · observed space only');
    if (view.robot && view.robot.age_s > 1) note.textContent += ' · telemetry stale';
    names.textContent = view.places.length ? 'Places: '+view.places.map(p=>p.name).join(', ') : 'No saved places yet';
    if ((view.rooms || []).length) names.textContent += ' · Rooms: '+view.rooms.map(p=>p.name.replace(/_/g,' ')).join(', ');
  }
  async function refresh() {
    try {
      const response = await fetch('/api/map', {cache:'no-store'});
      if (!response.ok) throw new Error('Map unavailable');
      draw(await response.json());
    } catch (error) { note.textContent = 'Map unavailable · last view may be stale'; }
    setTimeout(refresh, 1000);
  }
  refresh();
})();
