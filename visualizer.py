import cv2
import json
import numpy as np
import os

class Visualizer:
    def __init__(self, output_path, width, height, fps):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # FIX: Store web_assets inside the same directory as the HTML output (data/)
        # so the browser can resolve relative paths correctly: web_assets/det_X_Y.jpg
        output_dir = os.path.dirname(os.path.abspath(output_path))
        self.assets_dir = os.path.join(output_dir, "web_assets")
        if not os.path.exists(self.assets_dir):
            os.makedirs(self.assets_dir)

        # Muted, eye-friendly colors (BGR)
        self.colors = {
            'safe':     (144, 190, 109),  # olive green
            'moderate': (114, 158, 244),  # soft orange
            'danger':   (112, 108, 242)   # coral red
        }

    def process(self, frame, detections, gps_point, decision, frame_index):
        annotated = frame.copy()

        for i, d in enumerate(detections):
            x1, y1, x2, y2 = map(int, d.bbox)
            color = self.colors.get(d.risk_level, (200, 200, 200))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # FIX: Save only the cropped detection region instead of the full frame,
            # so each thumbnail shows the specific defect, not the entire scene.
            img_name = f"det_{frame_index}_{i}.jpg"
            img_path = os.path.join(self.assets_dir, img_name)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:  # Guard against zero-size crops from edge detections
                cv2.imwrite(img_path, crop)

            # Link the thumbnail filename to the detection object
            d.thumb_path = img_name

            label = f"{d.cls_name} | {d.risk_score}%"
            cv2.putText(annotated, label, (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        self._draw_info_panel(annotated, gps_point, decision, detections)
        self.out.write(annotated)
        return annotated

    def _draw_info_panel(self, frame, gps, decision, detections=None):
        h, w = frame.shape[:2]
        scale = w / 1280
        
        # MODIFIED: Increased width (from 0.35 to 0.42) and height (from 180 to 220) to fit text perfectly
        panel_w = int(w * 0.42)
        panel_h = int(220 * scale)
        
        overlay = frame.copy()
        
        # MODIFIED: Changed color from White (255, 255, 255) to Transparent Dark Gray (40, 40, 40)
        cv2.rectangle(overlay, (20, 20), (panel_w, panel_h), (40, 40, 40), -1)
        
        # MODIFIED: Adjusted blending weights (0.65 overlay, 0.35 frame) to achieve a balanced transparent gray look
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # MODIFIED: Changed text color to White (245, 245, 245) for clean contrast against the gray background
        text_color = (245, 245, 245)
        font = cv2.FONT_HERSHEY_SIMPLEX
        lines = [
            f"LAT/LNG: {gps['lat']:.5f}, {gps['lng']:.5f}",
            f"TIMESTAMP: {gps['simulated_date']}",
            f"AUTHORITY: {decision.party}"
        ]
        for i, line in enumerate(lines):
            y = int(65 * scale) + (i * int(50 * scale))  # Adjusted vertical spacing for the larger panel
            cv2.putText(frame, line, (45, y), font, 0.55 * scale, text_color, 1, cv2.LINE_AA)

    def generate_html_dashboard(self, all_detections_data, output_path):
        data_json = json.dumps(all_detections_data, ensure_ascii=False)

        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MARASD - Advanced Road GIS</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --danger:   #c0392b;
            --moderate: #d4830a;
            --safe:     #27764a;
            --danger-bg:   #fdf2f1;
            --moderate-bg: #fdf6ec;
            --safe-bg:     #f0f7f3;
            --bg:       #f5f4f0;
            --surface:  #ffffff;
            --border:   #e2e0da;
            --text:     #1a1916;
            --muted:    #7a7870;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            background: var(--bg);
            color: var(--text);
            font-family: 'DM Sans', sans-serif;
            min-height: 100vh;
        }}

        /* ── Header ── */
        .header {{
            padding: 16px 28px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: var(--surface);
        }}
        .header h1 {{
            font-size: 17px;
            font-weight: 600;
            letter-spacing: -0.2px;
            color: var(--text);
        }}
        .header h1 span {{ color: var(--danger); }}
        .header-sub {{ font-family: 'DM Mono', monospace; font-size: 11px; color: var(--muted); margin-top: 2px; }}

        /* ── KPI bar ── */
        .kpi-bar {{
            display: flex;
            gap: 0;
            border-bottom: 1px solid var(--border);
            background: var(--surface);
        }}
        .kpi {{
            flex: 1;
            padding: 13px 20px;
            display: flex;
            flex-direction: column;
            gap: 3px;
            border-right: 1px solid var(--border);
        }}
        .kpi:last-child {{ border-right: none; }}
        .kpi-label {{ font-size: 11px; color: var(--muted); font-weight: 500; }}
        .kpi-value {{ font-size: 22px; font-weight: 600; line-height: 1; }}
        .kpi-value.danger   {{ color: var(--danger); }}
        .kpi-value.moderate {{ color: var(--moderate); }}
        .kpi-value.safe     {{ color: var(--safe); }}
        .kpi-value.total    {{ color: var(--text); }}

        /* ── Main layout ── */
        .main {{
            display: grid;
            grid-template-columns: 1fr 320px;
            height: calc(100vh - 108px);
        }}

        /* ── Map ── */
        #map {{ width: 100%; height: 100%; }}

        /* ── Right sidebar ── */
        .sidebar {{
            background: var(--surface);
            border-left: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}

        .sidebar-tabs {{
            display: flex;
            border-bottom: 1px solid var(--border);
        }}
        .tab {{
            flex: 1;
            padding: 11px;
            text-align: center;
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            color: var(--muted);
            border-bottom: 2px solid transparent;
            transition: all 0.15s;
            background: none;
        }}
        .tab.active {{
            color: var(--text);
            border-bottom-color: var(--text);
        }}

        .tab-content {{ display: none; flex: 1; overflow-y: auto; padding: 16px; }}
        .tab-content.active {{ display: block; }}

        /* ── Analytics ── */
        .section-title {{
            font-size: 11px;
            color: var(--muted);
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            margin-bottom: 10px;
        }}

        .donut-wrap {{
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 20px;
        }}
        .donut-wrap svg {{ flex-shrink: 0; }}
        .donut-legend {{ display: flex; flex-direction: column; gap: 7px; }}
        .legend-row {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
        }}
        .legend-dot {{ width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }}
        .legend-pct {{ font-family: 'DM Mono', monospace; font-size: 12px; margin-left: auto; font-weight: 500; }}

        /* Horizontal bar chart */
        .bar-group {{ margin-bottom: 20px; }}
        .bar-row {{ margin-bottom: 10px; }}
        .bar-meta {{ display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px; }}
        .bar-meta span:last-child {{ font-family: 'DM Mono', monospace; font-weight: 500; }}
        .bar-track {{ background: #ebe9e4; border-radius: 2px; height: 6px; overflow: hidden; }}
        .bar-fill {{ height: 100%; border-radius: 2px; transition: width 0.5s ease; }}

        /* Divider */
        .divider {{ height: 1px; background: var(--border); margin: 14px 0; }}

        /* Class breakdown table */
        .breakdown-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
        .breakdown-table th {{
            font-size: 11px;
            color: var(--muted);
            font-weight: 500;
            text-align: left;
            padding: 0 0 7px 0;
            border-bottom: 1px solid var(--border);
        }}
        .breakdown-table td {{
            padding: 7px 0;
            border-bottom: 1px solid var(--border);
            vertical-align: middle;
        }}
        .breakdown-table tr:last-child td {{ border-bottom: none; }}

        /* Risk badge */
        .risk-badge {{
            display: inline-block;
            padding: 2px 7px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 600;
            font-family: 'DM Mono', monospace;
        }}

        /* ── Priority list ── */
        .priority-item {{
            padding: 10px 8px;
            border-bottom: 1px solid var(--border);
            cursor: pointer;
            transition: background 0.12s;
            border-radius: 6px;
        }}
        .priority-item:hover {{ background: #f0ede8; }}
        .priority-item:last-child {{ border-bottom: none; }}
        .pi-top {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px; }}
        .pi-class {{ font-size: 13px; font-weight: 600; text-transform: capitalize; }}
        .pi-sub {{ font-size: 11px; color: var(--muted); font-family: 'DM Mono', monospace; }}

        /* Scrollbar */
        .tab-content::-webkit-scrollbar {{ width: 4px; }}
        .tab-content::-webkit-scrollbar-track {{ background: transparent; }}
        .tab-content::-webkit-scrollbar-thumb {{ background: #d0cec8; border-radius: 2px; }}
    </style>
</head>
<body>

    <div class="header">
        <div>
            <h1>MARASD <span>Road Intelligence</span></h1>
            <div class="header-sub">AUTOMATED ROAD DEFECT DETECTION &amp; GIS ANALYSIS</div>
        </div>
    </div>

    <div class="kpi-bar">
        <div class="kpi"><div class="kpi-label">Total Defects</div><div class="kpi-value total" id="kpi-total">—</div></div>
        <div class="kpi"><div class="kpi-label">Critical</div><div class="kpi-value danger" id="kpi-danger">—</div></div>
        <div class="kpi"><div class="kpi-label">Moderate</div><div class="kpi-value moderate" id="kpi-moderate">—</div></div>
        <div class="kpi"><div class="kpi-label">Low Risk</div><div class="kpi-value safe" id="kpi-safe">—</div></div>
        <div class="kpi"><div class="kpi-label">Avg Risk Score</div><div class="kpi-value total" id="kpi-avg">—</div></div>
    </div>

    <div class="main">
        <div id="map"></div>

        <div class="sidebar">
            <div class="sidebar-tabs">
                <div class="tab active" onclick="switchTab('analytics', this)">Analytics</div>
                <div class="tab" onclick="switchTab('list', this)">Priority List</div>
            </div>

            <div class="tab-content active" id="tab-analytics">

                <div class="section-title">Distribution by Severity</div>
                <div class="donut-wrap">
                    <svg id="donut" width="110" height="110" viewBox="0 0 110 110"></svg>
                    <div class="donut-legend" id="donut-legend"></div>
                </div>

                <div class="divider"></div>

                <div class="section-title">Risk Score by Class</div>
                <div class="bar-group" id="class-bars"></div>

                <div class="divider"></div>

                <div class="section-title">Defect Breakdown</div>
                <table class="breakdown-table">
                    <thead>
                        <tr>
                            <th>Class</th>
                            <th>Count</th>
                            <th>Avg Score</th>
                            <th>Level</th>
                        </tr>
                    </thead>
                    <tbody id="breakdown-body"></tbody>
                </table>
            </div>

            <div class="tab-content" id="tab-list">
                <div id="priority-list"></div>
            </div>
        </div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        const DATA   = {data_json};
        const COLORS    = {{ danger: "#c0392b", moderate: "#d4830a", safe: "#27764a" }};
        const COLORS_BG = {{ danger: "#fdf2f1", moderate: "#fdf6ec", safe: "#f0f7f3" }};

        // ── Tab switcher ──────────────────────────────────────────────────
        function switchTab(id, el) {{
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            el.classList.add('active');
            document.getElementById('tab-' + id).classList.add('active');
        }}

        // ── Map ───────────────────────────────────────────────────────────
        const map = L.map("map", {{ zoomControl: false }})
            .setView([DATA[0]?.lat || 24.7136, DATA[0]?.lng || 46.6753], 14);

        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            attribution: '&copy; OpenStreetMap &copy; CARTO'
        }}).addTo(map);

        L.control.zoom({{ position: 'bottomright' }}).addTo(map);

        DATA.forEach(d => {{
            const marker = L.circleMarker([d.lat, d.lng], {{
                radius: 8, color: '#fff', weight: 1.5,
                fillColor: COLORS[d.risk_level], fillOpacity: 0.9
            }}).addTo(map);

            // FIX: Image path is now relative to the HTML file location.
            // Both the HTML and web_assets/ folder live inside data/, so this resolves correctly.
            const popupHTML = `
                <img src="web_assets/${{d.thumb_path}}" class="popup-img"
                     style="width:100%;height:180px;object-fit:cover;display:block;">
                <div style="padding:14px;background:#ffffff;color:#1a1916;font-family:'DM Sans',sans-serif;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                        <strong style="font-size:14px;text-transform:capitalize">${{d.class}}</strong>
                        <span style="background:${{COLORS_BG[d.risk_level]}};color:${{COLORS[d.risk_level]}};padding:3px 8px;border-radius:4px;font-size:11px;font-weight:600;">${{d.risk_score}}%</span>
                    </div>
                    <div style="font-size:11px;color:#7a7870;font-family:'DM Mono',monospace;line-height:1.8;">
                        <div>AUTHORITY: ${{d.party}}</div>
                        <div>DEPTH: ${{d.depth || 'N/A'}} cm</div>
                        <div>GPS: ${{d.lat.toFixed(5)}}, ${{d.lng.toFixed(5)}}</div>
                    </div>
                </div>`;

            marker.bindPopup(popupHTML, {{ maxWidth: 320 }});
        }});

        // ── KPI cards ─────────────────────────────────────────────────────
        const counts  = {{ danger: 0, moderate: 0, safe: 0 }};
        let totalScore = 0;
        DATA.forEach(d => {{
            if (counts[d.risk_level] !== undefined) counts[d.risk_level]++;
            totalScore += (d.risk_score || 0);
        }});
        document.getElementById('kpi-total').textContent    = DATA.length;
        document.getElementById('kpi-danger').textContent   = counts.danger;
        document.getElementById('kpi-moderate').textContent = counts.moderate;
        document.getElementById('kpi-safe').textContent     = counts.safe;
        document.getElementById('kpi-avg').textContent      = DATA.length
            ? Math.round(totalScore / DATA.length) + '%' : '—';

        // ── Donut chart ───────────────────────────────────────────────────
        function drawDonut() {{
            const svg    = document.getElementById('donut');
            const legend = document.getElementById('donut-legend');
            const cx = 55, cy = 55, r = 40, stroke = 14;
            const total  = DATA.length || 1;
            const slices = [
                {{ key: 'danger',   label: 'Critical', color: COLORS.danger }},
                {{ key: 'moderate', label: 'Moderate', color: COLORS.moderate }},
                {{ key: 'safe',     label: 'Low Risk', color: COLORS.safe }},
            ];

            const circ = 2 * Math.PI * r;
            let offset = 0;
            svg.innerHTML = '';

            // Background ring
            const bg = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            bg.setAttribute('cx', cx); bg.setAttribute('cy', cy); bg.setAttribute('r', r);
            bg.setAttribute('fill', 'none'); bg.setAttribute('stroke', '#ebe9e4');
            bg.setAttribute('stroke-width', stroke);
            svg.appendChild(bg);

            slices.forEach(s => {{
                const pct  = counts[s.key] / total;
                const dash = pct * circ;
                const gap  = circ - dash;
                const arc  = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                arc.setAttribute('cx', cx); arc.setAttribute('cy', cy); arc.setAttribute('r', r);
                arc.setAttribute('fill', 'none');
                arc.setAttribute('stroke', s.color);
                arc.setAttribute('stroke-width', stroke);
                arc.setAttribute('stroke-dasharray', `${{dash}} ${{gap}}`);
                arc.setAttribute('stroke-dashoffset', -offset * circ);
                arc.setAttribute('transform', `rotate(-90 ${{cx}} ${{cy}})`);
                svg.appendChild(arc);

                const pctText = Math.round(pct * 100);
                legend.innerHTML += `
                    <div class="legend-row">
                        <div class="legend-dot" style="background:${{s.color}}"></div>
                        <span style="font-size:12px">${{s.label}}</span>
                        <span class="legend-pct" style="color:${{s.color}}">${{pctText}}%</span>
                    </div>`;

                offset += pct;
            }});

            // Center label showing total defect count
            const txt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            txt.setAttribute('x', cx); txt.setAttribute('y', cy + 5);
            txt.setAttribute('text-anchor', 'middle');
            txt.setAttribute('fill', '#1a1916');
            txt.setAttribute('font-size', '15');
            txt.setAttribute('font-weight', '600');
            txt.setAttribute('font-family', 'DM Sans, sans-serif');
            txt.textContent = DATA.length;
            svg.appendChild(txt);
        }}
        drawDonut();

        // ── Bar chart: avg risk score by class ────────────────────────────
        function drawClassBars() {{
            const classMap = {{}};
            DATA.forEach(d => {{
                if (!classMap[d.class]) classMap[d.class] = {{ total: 0, count: 0, level: d.risk_level }};
                classMap[d.class].total += (d.risk_score || 0);
                classMap[d.class].count++;
            }});

            const container = document.getElementById('class-bars');
            container.innerHTML = '';

            // Sort classes by average risk score descending
            const sorted = Object.entries(classMap).sort((a, b) =>
                (b[1].total / b[1].count) - (a[1].total / a[1].count));

            sorted.forEach(([cls, val]) => {{
                const avg   = Math.round(val.total / val.count);
                const color = COLORS[val.level] || '#6b7280';
                container.innerHTML += `
                    <div class="bar-row">
                        <div class="bar-meta">
                            <span style="text-transform:capitalize;font-size:12px">${{cls}}</span>
                            <span style="color:${{color}}">${{avg}}%</span>
                        </div>
                        <div class="bar-track">
                            <div class="bar-fill" style="width:${{avg}}%;background:${{color}}"></div>
                        </div>
                    </div>`;
            }});

            // Breakdown table
            const tbody = document.getElementById('breakdown-body');
            sorted.forEach(([cls, val]) => {{
                const avg   = Math.round(val.total / val.count);
                const color = COLORS[val.level] || '#6b7280';
                const levelLabel = val.level.charAt(0).toUpperCase() + val.level.slice(1);
                tbody.innerHTML += `
                    <tr>
                        <td style="text-transform:capitalize;font-size:12px">${{cls}}</td>
                        <td style="font-family:'DM Mono',monospace;font-size:12px">${{val.count}}</td>
                        <td style="font-family:'DM Mono',monospace;font-size:12px;color:${{color}}">${{avg}}%</td>
                        <td><span class="risk-badge" style="background:${{COLORS_BG[val.level]}};color:${{color}}">${{levelLabel}}</span></td>
                    </tr>`;
            }});
        }}
        drawClassBars();

        // ── Priority list: sorted by risk score (highest first) ───────────
        function buildList() {{
            const sorted = [...DATA].sort((a, b) => (b.risk_score || 0) - (a.risk_score || 0));
            const el = document.getElementById('priority-list');
            sorted.forEach(d => {{
                const color = COLORS[d.risk_level] || '#6b7280';
                el.innerHTML += `
                    <div class="priority-item" onclick="map.flyTo([${{d.lat}}, ${{d.lng}}], 18)">
                        <div class="pi-top">
                            <span class="pi-class">${{d.class}}</span>
                            <span class="risk-badge" style="background:${{COLORS_BG[d.risk_level]}};color:${{COLORS[d.risk_level]}}">${{d.risk_score}}%</span>
                        </div>
                        <div class="pi-sub">${{d.party}} · ${{d.lat.toFixed(4)}}, ${{d.lng.toFixed(4)}}</div>
                    </div>`;
            }});
        }}
        buildList();
    </script>
</body>
</html>
"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_template)

    def release(self):
        if self.out:
            self.out.release()