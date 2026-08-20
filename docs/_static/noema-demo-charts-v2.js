(function () {
  "use strict";

  const CHART_SELECTOR = "[data-noema-chart]";
  const MIN_VIEW_FRACTION = 0.035;
  const MAX_VIEW_MULTIPLIER = 8;

  function finite(value) {
    return Number.isFinite(Number(value));
  }

  function clamp(value, minimum, maximum) {
    return Math.max(minimum, Math.min(maximum, value));
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function slug(value) {
    return String(value || "chart")
      .toLowerCase()
      .replace(/[^a-z0-9_-]+/g, "-")
      .replace(/^-+|-+$/g, "") || "chart";
  }

  function chartTheme(root) {
    const style = getComputedStyle(root);
    const read = (name, fallback) => style.getPropertyValue(name).trim() || fallback;
    return {
      text: read("--noema-chart-text", "#111827"),
      muted: read("--noema-chart-muted", "#64748b"),
      grid: read("--noema-chart-grid", "rgba(100, 116, 139, 0.2)"),
      axis: read("--noema-chart-axis", "#64748b"),
      background: read("--noema-chart-background", "#ffffff"),
      tooltipBackground: read("--noema-chart-tooltip-background", "#0f172a"),
      tooltipText: read("--noema-chart-tooltip-text", "#f8fafc"),
    };
  }

  function formatValue(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    const magnitude = Math.abs(number);
    if (magnitude !== 0 && (magnitude < 0.001 || magnitude >= 10000)) {
      return number.toExponential(2);
    }
    if (magnitude >= 100) return number.toFixed(0);
    if (magnitude >= 10) return number.toFixed(1).replace(/\.0$/, "");
    if (magnitude >= 1) return number.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
    return number.toPrecision(3).replace(/0+$/, "").replace(/\.$/, "");
  }

  function axisTicks(minimum, maximum, count) {
    if (!(maximum > minimum)) return [minimum];
    const rough = (maximum - minimum) / Math.max(1, count - 1);
    const magnitude = 10 ** Math.floor(Math.log10(rough));
    const normalized = rough / magnitude;
    const step = (normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10) * magnitude;
    const first = Math.ceil(minimum / step) * step;
    const ticks = [];
    for (let value = first; value <= maximum + step * 1e-8; value += step) {
      ticks.push(Number(value.toPrecision(12)));
    }
    return ticks;
  }

  class NoemaDemoChart {
    constructor(root, spec) {
      this.root = root;
      this.spec = spec;
      this.hidden = new Set();
      this.highlighted = "";
      this.yScale = spec.yScale === "log" ? "log" : "linear";
      this.view = null;
      this.drag = null;
      this.plot = null;
      this.theme = chartTheme(root);
      this.boundarySegments = new Map();
      this.accessibleSummary = (
        root.dataset.chartSummary
        || spec.accessibleSummary
        || spec.description
        || spec.title
      );
      this.renderShell();
      this.bind();
      this.resetView();
      this.resize();
    }

    renderShell() {
      const scaleLabel = this.hasRightAxis()
        ? (this.yScale === "log" ? "Log channel axis" : "Linear channel axis")
        : (this.yScale === "log" ? "Log Y" : "Linear Y");
      const logControl = this.spec.type === "line" && this.spec.allowLog
        ? `<button type="button" class="noema-chart-control" data-chart-action="scale" aria-pressed="${this.yScale === "log"}">${scaleLabel}</button>`
        : "";
      const chartId = slug(this.root.dataset.noemaChart);
      const titleId = `noema-chart-${chartId}-title`;
      const descriptionId = `noema-chart-${chartId}-description`;
      const summaryId = `noema-chart-${chartId}-summary`;
      this.root.classList.add("noema-doc-chart");
      this.root.setAttribute("role", "figure");
      this.root.setAttribute("aria-labelledby", titleId);
      this.root.innerHTML = `
        <div class="noema-chart-heading">
          <div>
            <h3 class="noema-chart-title" id="${titleId}">${escapeHtml(this.spec.title)}</h3>
            <p class="noema-chart-description" id="${descriptionId}">${escapeHtml(this.spec.description || "")}</p>
          </div>
          <div class="noema-chart-controls" role="group" aria-label="${escapeHtml(this.spec.title)} controls">
            ${logControl}
            <button type="button" class="noema-chart-control" data-chart-action="reset">Reset view</button>
          </div>
        </div>
        <ul class="noema-chart-legend" aria-label="${escapeHtml(this.spec.title)} methods"></ul>
        <div class="noema-chart-stage">
          <canvas class="noema-chart-canvas" role="img" aria-labelledby="${titleId}" aria-describedby="${descriptionId} ${summaryId}"></canvas>
          <div class="noema-chart-tooltip" role="status" aria-live="polite" hidden></div>
        </div>
        <p class="noema-chart-accessible-summary" id="${summaryId}">${escapeHtml(this.accessibleSummary)} Exact means are in the table below; downloadable chart data includes interval bounds.</p>
        <div class="noema-chart-interaction-hint">
          <span class="noema-chart-hint-fine">Wheel to zoom · drag to pan · double-click to reset</span>
          <span class="noema-chart-hint-coarse">Drag horizontally to pan · use Reset view</span>
        </div>
      `;
      this.canvas = this.root.querySelector("canvas");
      this.context = this.canvas.getContext("2d");
      this.tooltip = this.root.querySelector(".noema-chart-tooltip");
      this.legend = this.root.querySelector(".noema-chart-legend");
      this.renderLegend();
    }

    renderLegend() {
      this.legend.innerHTML = (this.spec.series || []).map((series) => `
        <li>
          <button
            type="button"
            class="noema-chart-legend-item"
            data-series-id="${escapeHtml(series.id)}"
            data-marker="${escapeHtml(series.marker || "circle")}"
            title="${escapeHtml(series.label)}"
            aria-label="${escapeHtml(`${series.label}; select to show or hide`)}"
            aria-pressed="true"
            style="--series-color:${escapeHtml(series.color)}"
          >
            <span class="noema-chart-legend-line${series.dash && series.dash.length ? " dashed" : ""}" aria-hidden="true">
              <span class="noema-chart-legend-marker"></span>
            </span>
            <span>${escapeHtml(series.label)}</span>
          </button>
        </li>
      `).join("");
    }

    bind() {
      this.root.querySelectorAll("[data-chart-action]").forEach((button) => {
        button.addEventListener("click", () => {
          if (button.dataset.chartAction === "reset") {
            this.resetView();
            return;
          }
          if (button.dataset.chartAction === "scale") {
            this.yScale = this.yScale === "log" ? "linear" : "log";
            button.textContent = this.hasRightAxis()
              ? (this.yScale === "log" ? "Log channel axis" : "Linear channel axis")
              : (this.yScale === "log" ? "Log Y" : "Linear Y");
            button.setAttribute("aria-pressed", String(this.yScale === "log"));
            this.resetView();
          }
        });
      });
      this.legend.querySelectorAll("[data-series-id]").forEach((button) => {
        const id = button.dataset.seriesId;
        const highlight = () => {
          this.highlighted = id;
          this.draw();
        };
        const clear = () => {
          if (this.highlighted === id) {
            this.highlighted = "";
            this.draw();
          }
        };
        button.addEventListener("mouseenter", highlight);
        button.addEventListener("focus", highlight);
        button.addEventListener("mouseleave", clear);
        button.addEventListener("blur", clear);
        button.addEventListener("click", () => {
          if (this.hidden.has(id)) this.hidden.delete(id);
          else if (this.spec.series.length - this.hidden.size > 1) this.hidden.add(id);
          button.classList.toggle("is-hidden", this.hidden.has(id));
          button.setAttribute("aria-pressed", String(!this.hidden.has(id)));
          this.draw();
        });
      });
      this.canvas.addEventListener("wheel", (event) => this.onWheel(event), { passive: false });
      this.canvas.addEventListener("pointerdown", (event) => this.onPointerDown(event));
      this.canvas.addEventListener("pointermove", (event) => this.onPointerMove(event));
      this.canvas.addEventListener("pointerup", (event) => this.onPointerUp(event));
      this.canvas.addEventListener("pointercancel", (event) => this.onPointerUp(event));
      this.canvas.addEventListener("pointerleave", () => {
        if (!this.drag) this.hideTooltip();
      });
      this.canvas.addEventListener("dblclick", (event) => {
        event.preventDefault();
        this.resetView();
      });
      this.resizeObserver = new ResizeObserver(() => this.resize());
      this.resizeObserver.observe(this.root);
    }

    dataDomain() {
      if (this.spec.type === "decision-boundary") {
        return {
          xMin: Number(this.spec.xDomain[0]),
          xMax: Number(this.spec.xDomain[1]),
          yMin: Number(this.spec.yDomain[0]),
          yMax: Number(this.spec.yDomain[1]),
        };
      }
      const visible = (this.spec.series || []).filter((series) => !this.hidden.has(series.id));
      const points = visible.flatMap((series) => series.values || []).filter((point) => finite(point[0]) && finite(point[1]));
      const ranges = visible.flatMap((series) => series.range || []).filter((point) => finite(point[0]) && finite(point[1]) && finite(point[2]));
      const xs = points.map((point) => Number(point[0])).concat(ranges.map((point) => Number(point[0])));
      let xMin = Math.min(...xs);
      let xMax = Math.max(...xs);
      const xPad = Math.max(1e-9, (xMax - xMin) * 0.055);
      xMin -= xPad;
      xMax += xPad;
      const axisDomain = (axis, scale, includeZero) => {
        let axisSeries = visible.filter((series) => this.seriesAxis(series) === axis);
        if (!axisSeries.length) {
          axisSeries = (this.spec.series || []).filter((series) => this.seriesAxis(series) === axis);
        }
        let ys = axisSeries
          .flatMap((series) => (series.values || []).map((point) => point[1])
            .concat((series.range || []).flatMap((point) => [point[1], point[2]])))
          .filter(finite)
          .map(Number);
        if (scale === "log") ys = ys.filter((value) => value > 0);
        if (!ys.length) return { min: 0, max: 1 };
        if (scale === "log") {
          return {
            min: Math.log10(Math.min(...ys)) - 0.14,
            max: Math.log10(Math.max(...ys)) + 0.12,
          };
        }
        const minimum = Math.min(...ys);
        const maximum = Math.max(...ys);
        const pad = Math.max(1e-12, (maximum - minimum) * 0.1);
        return {
          min: includeZero ? Math.min(0, minimum - pad) : minimum - pad,
          max: includeZero ? Math.max(0, maximum + pad) : maximum + pad,
        };
      };
      const left = axisDomain("left", this.yScale, this.spec.yIncludeZero);
      if (
        Array.isArray(this.spec.yDomain)
        && this.spec.yDomain.length === 2
        && finite(this.spec.yDomain[0])
        && finite(this.spec.yDomain[1])
      ) {
        const requestedMinimum = Number(this.spec.yDomain[0]);
        const requestedMaximum = Number(this.spec.yDomain[1]);
        if (
          requestedMaximum > requestedMinimum
          && (this.yScale !== "log" || requestedMinimum > 0)
        ) {
          left.min = this.yScale === "log"
            ? Math.log10(requestedMinimum)
            : requestedMinimum;
          left.max = this.yScale === "log"
            ? Math.log10(requestedMaximum)
            : requestedMaximum;
        }
      }
      const domain = { xMin, xMax, yMin: left.min, yMax: left.max };
      if (this.hasRightAxis()) {
        const right = axisDomain(
          "right",
          this.spec.rightYScale === "log" ? "log" : "linear",
          this.spec.rightYIncludeZero,
        );
        domain.rightYMin = right.min;
        domain.rightYMax = right.max;
      }
      return domain;
    }

    resetView() {
      this.view = this.dataDomain();
      this.hideTooltip();
      this.draw();
    }

    resize() {
      const stage = this.root.querySelector(".noema-chart-stage");
      const width = Math.max(320, Math.floor(stage.clientWidth || this.root.clientWidth || 720));
      const height = this.spec.type === "decision-boundary"
        ? clamp(Math.round(width * 0.64), 360, 560)
        : clamp(Math.round(width * 0.56), 340, 500);
      const ratio = clamp(window.devicePixelRatio || 1, 1, 2);
      this.canvas.style.height = `${height}px`;
      this.canvas.width = Math.round(width * ratio);
      this.canvas.height = Math.round(height * ratio);
      this.canvas.style.width = `${width}px`;
      this.context.setTransform(ratio, 0, 0, ratio, 0, 0);
      this.width = width;
      this.height = height;
      this.draw();
    }

    plotBounds() {
      return {
        left: 76,
        right: this.width - (this.hasRightAxis() ? 76 : 24),
        top: 20,
        bottom: this.height - 54,
      };
    }

    xToPixel(value) {
      return this.plot.left + (Number(value) - this.view.xMin) / (this.view.xMax - this.view.xMin) * (this.plot.right - this.plot.left);
    }

    hasRightAxis() {
      return Boolean(
        this.spec.rightYLabel
        && (this.spec.series || []).some((series) => this.seriesAxis(series) === "right"),
      );
    }

    seriesAxis(series) {
      return series && series.yAxis === "right" ? "right" : "left";
    }

    axisScale(axis) {
      if (axis === "right") return this.spec.rightYScale === "log" ? "log" : "linear";
      return this.yScale;
    }

    yTransform(value, axis = "left") {
      return this.axisScale(axis) === "log" ? Math.log10(Number(value)) : Number(value);
    }

    yToPixel(value, axis = "left") {
      const transformed = this.spec.type === "line" ? this.yTransform(value, axis) : Number(value);
      const minimum = axis === "right" ? this.view.rightYMin : this.view.yMin;
      const maximum = axis === "right" ? this.view.rightYMax : this.view.yMax;
      return this.plot.bottom - (transformed - minimum) / (maximum - minimum) * (this.plot.bottom - this.plot.top);
    }

    pixelToX(pixel) {
      return this.view.xMin + (pixel - this.plot.left) / (this.plot.right - this.plot.left) * (this.view.xMax - this.view.xMin);
    }

    pixelToY(pixel, axis = "left") {
      const minimum = axis === "right" ? this.view.rightYMin : this.view.yMin;
      const maximum = axis === "right" ? this.view.rightYMax : this.view.yMax;
      const transformed = maximum - (pixel - this.plot.top) / (this.plot.bottom - this.plot.top) * (maximum - minimum);
      return this.spec.type === "line" && this.axisScale(axis) === "log" ? 10 ** transformed : transformed;
    }

    draw() {
      if (!this.context || !this.width || !this.view) return;
      this.theme = chartTheme(this.root);
      this.plot = this.plotBounds();
      const context = this.context;
      context.clearRect(0, 0, this.width, this.height);
      context.fillStyle = this.theme.background;
      context.fillRect(0, 0, this.width, this.height);
      this.drawAxes();
      context.save();
      context.beginPath();
      context.rect(this.plot.left, this.plot.top, this.plot.right - this.plot.left, this.plot.bottom - this.plot.top);
      context.clip();
      if (this.spec.type === "decision-boundary") this.drawBoundaries();
      else this.drawLines();
      context.restore();
    }

    drawAxes() {
      const context = this.context;
      const xTicks = axisTicks(this.view.xMin, this.view.xMax, 6);
      const yTicks = axisTicks(this.view.yMin, this.view.yMax, 6);
      context.font = "12px system-ui, sans-serif";
      context.lineWidth = 1;
      context.textBaseline = "middle";
      xTicks.forEach((tick) => {
        const x = this.xToPixel(tick);
        context.strokeStyle = this.theme.grid;
        context.beginPath();
        context.moveTo(x, this.plot.top);
        context.lineTo(x, this.plot.bottom);
        context.stroke();
        context.fillStyle = this.theme.muted;
        context.textAlign = "center";
        context.fillText(formatValue(tick), x, this.plot.bottom + 19);
      });
      yTicks.forEach((tick) => {
        const value = this.spec.type === "line" && this.yScale === "log" ? 10 ** tick : tick;
        const y = this.spec.type === "line" && this.yScale === "log"
          ? this.plot.bottom - (tick - this.view.yMin) / (this.view.yMax - this.view.yMin) * (this.plot.bottom - this.plot.top)
          : this.yToPixel(value);
        context.strokeStyle = this.theme.grid;
        context.beginPath();
        context.moveTo(this.plot.left, y);
        context.lineTo(this.plot.right, y);
        context.stroke();
        context.fillStyle = this.theme.muted;
        context.textAlign = "right";
        context.fillText(formatValue(value), this.plot.left - 9, y);
      });
      if (this.hasRightAxis()) {
        const rightScale = this.axisScale("right");
        const rightTicks = axisTicks(this.view.rightYMin, this.view.rightYMax, 6);
        rightTicks.forEach((tick) => {
          const value = rightScale === "log" ? 10 ** tick : tick;
          const y = rightScale === "log"
            ? this.plot.bottom - (tick - this.view.rightYMin) / (this.view.rightYMax - this.view.rightYMin) * (this.plot.bottom - this.plot.top)
            : this.yToPixel(value, "right");
          context.fillStyle = this.theme.muted;
          context.textAlign = "left";
          context.fillText(formatValue(value), this.plot.right + 9, y);
        });
      }
      context.strokeStyle = this.theme.axis;
      context.strokeRect(this.plot.left, this.plot.top, this.plot.right - this.plot.left, this.plot.bottom - this.plot.top);
      context.fillStyle = this.theme.text;
      context.textAlign = "center";
      context.textBaseline = "alphabetic";
      context.fillText(this.spec.xLabel, (this.plot.left + this.plot.right) / 2, this.height - 8);
      context.save();
      context.translate(17, (this.plot.top + this.plot.bottom) / 2);
      context.rotate(-Math.PI / 2);
      context.fillText(this.spec.yLabel, 0, 0);
      context.restore();
      if (this.hasRightAxis()) {
        context.save();
        context.translate(this.width - 17, (this.plot.top + this.plot.bottom) / 2);
        context.rotate(Math.PI / 2);
        context.fillText(this.spec.rightYLabel, 0, 0);
        context.restore();
      }
    }

    drawLines() {
      const context = this.context;
      (this.spec.series || []).forEach((series) => {
        if (this.hidden.has(series.id) || !Array.isArray(series.range)) return;
        const range = series.range.filter((point) => point[1] > 0 && point[2] > 0);
        if (!range.length) return;
        const axis = this.seriesAxis(series);
        context.globalAlpha = this.highlighted && this.highlighted !== series.id ? 0.05 : 0.13;
        context.fillStyle = series.color;
        context.beginPath();
        range.forEach((point, index) => {
          const x = this.xToPixel(point[0]);
          const y = this.yToPixel(point[2], axis);
          if (index) context.lineTo(x, y);
          else context.moveTo(x, y);
        });
        [...range].reverse().forEach((point) => context.lineTo(this.xToPixel(point[0]), this.yToPixel(point[1], axis)));
        context.closePath();
        context.fill();
      });
      (this.spec.series || []).forEach((series) => {
        if (this.hidden.has(series.id) || !Array.isArray(series.range)) return;
        const axis = this.seriesAxis(series);
        const ranges = series.range.filter(
          (point) => finite(point[0]) && finite(point[1]) && finite(point[2])
            && Number(point[2]) > Number(point[1]),
        );
        if (!ranges.length) return;
        context.save();
        context.globalAlpha = this.highlighted && this.highlighted !== series.id ? 0.1 : 0.72;
        context.strokeStyle = series.color;
        context.lineWidth = 1.25;
        context.setLineDash([]);
        ranges.forEach((point) => {
          const x = this.xToPixel(point[0]);
          const low = this.yToPixel(point[1], axis);
          const high = this.yToPixel(point[2], axis);
          context.beginPath();
          context.moveTo(x, low);
          context.lineTo(x, high);
          context.moveTo(x - 3.5, low);
          context.lineTo(x + 3.5, low);
          context.moveTo(x - 3.5, high);
          context.lineTo(x + 3.5, high);
          context.stroke();
        });
        context.restore();
      });
      (this.spec.series || []).forEach((series) => {
        if (this.hidden.has(series.id)) return;
        const axis = this.seriesAxis(series);
        const scale = this.axisScale(axis);
        const points = (series.values || []).filter((point) => finite(point[0]) && finite(point[1]) && (scale !== "log" || point[1] > 0));
        const highlighted = !this.highlighted || this.highlighted === series.id;
        context.globalAlpha = highlighted ? 1 : 0.16;
        context.strokeStyle = series.color;
        context.fillStyle = series.color;
        context.lineWidth = this.highlighted === series.id ? 3.4 : 2.2;
        context.setLineDash(series.dash || []);
        context.beginPath();
        points.forEach((point, index) => {
          const x = this.xToPixel(point[0]);
          const y = this.yToPixel(point[1], axis);
          if (index) context.lineTo(x, y);
          else context.moveTo(x, y);
        });
        context.stroke();
        context.setLineDash([]);
        points.forEach((point) => this.drawMarker(series.marker, this.xToPixel(point[0]), this.yToPixel(point[1], axis), series.color));
      });
      context.globalAlpha = 1;
    }

    drawMarker(kind, x, y, color) {
      const context = this.context;
      const size = 4.2;
      context.save();
      context.strokeStyle = color;
      context.fillStyle = this.theme.background;
      context.lineWidth = 2;
      context.beginPath();
      if (kind === "square") context.rect(x - size, y - size, size * 2, size * 2);
      else if (kind === "triangle") {
        context.moveTo(x, y - size - 1);
        context.lineTo(x + size + 1, y + size);
        context.lineTo(x - size - 1, y + size);
        context.closePath();
      } else if (kind === "diamond") {
        context.moveTo(x, y - size - 1);
        context.lineTo(x + size + 1, y);
        context.lineTo(x, y + size + 1);
        context.lineTo(x - size - 1, y);
        context.closePath();
      } else if (kind === "cross") {
        context.moveTo(x - size, y - size);
        context.lineTo(x + size, y + size);
        context.moveTo(x + size, y - size);
        context.lineTo(x - size, y + size);
        context.stroke();
        context.restore();
        return;
      } else context.arc(x, y, size, 0, Math.PI * 2);
      context.fill();
      context.stroke();
      context.restore();
    }

    segmentsFor(series) {
      if (this.boundarySegments.has(series.id)) return this.boundarySegments.get(series.id);
      const rows = series.classRows || [];
      const height = rows.length;
      const width = height ? rows[0].length : 0;
      const xMin = Number(this.spec.xDomain[0]);
      const xMax = Number(this.spec.xDomain[1]);
      const yMin = Number(this.spec.yDomain[0]);
      const yMax = Number(this.spec.yDomain[1]);
      const cellWidth = (xMax - xMin) / width;
      const cellHeight = (yMax - yMin) / height;
      const segments = [];
      rows.forEach((row, sourceRow) => {
        for (let column = 1; column < width; column += 1) {
          if (row[column] !== row[column - 1]) {
            const x = xMin + column * cellWidth;
            // classRows are stored from q_min to q_max. These are data
            // coordinates; yToPixel performs the only screen-axis inversion.
            const y = yMin + sourceRow * cellHeight;
            segments.push([x, y, x, y + cellHeight]);
          }
        }
      });
      for (let sourceRow = 1; sourceRow < height; sourceRow += 1) {
        const previous = rows[sourceRow - 1];
        const current = rows[sourceRow];
        const y = yMin + sourceRow * cellHeight;
        for (let column = 0; column < width; column += 1) {
          if (current[column] !== previous[column]) {
            const x = xMin + column * cellWidth;
            segments.push([x, y, x + cellWidth, y]);
          }
        }
      }
      this.boundarySegments.set(series.id, segments);
      return segments;
    }

    drawBoundaries() {
      const context = this.context;
      (this.spec.series || []).forEach((series) => {
        if (this.hidden.has(series.id)) return;
        context.globalAlpha = !this.highlighted || this.highlighted === series.id ? 0.96 : 0.14;
        context.strokeStyle = series.color;
        context.lineWidth = this.highlighted === series.id ? 3.5 : 2.2;
        context.setLineDash(series.dash || []);
        context.beginPath();
        this.segmentsFor(series).forEach((segment) => {
          context.moveTo(this.xToPixel(segment[0]), this.yToPixel(segment[1]));
          context.lineTo(this.xToPixel(segment[2]), this.yToPixel(segment[3]));
        });
        context.stroke();
      });
      context.globalAlpha = 1;
      context.setLineDash([]);
      const constellation = this.spec.constellation || [];
      const center = constellation.reduce(
        (sum, point) => ({
          i: sum.i + Number(point.i) / Math.max(1, constellation.length),
          q: sum.q + Number(point.q) / Math.max(1, constellation.length),
        }),
        { i: 0, q: 0 },
      );
      constellation.forEach((point) => {
        const x = this.xToPixel(point.i);
        const y = this.yToPixel(point.q);
        const labelRight = Number(point.i) >= center.i;
        const labelAbove = Number(point.q) >= center.q;
        context.fillStyle = this.theme.background;
        context.strokeStyle = this.theme.text;
        context.lineWidth = 1.7;
        context.beginPath();
        context.arc(x, y, 4.5, 0, Math.PI * 2);
        context.fill();
        context.stroke();
        context.fillStyle = this.theme.text;
        context.font = "11px system-ui, sans-serif";
        context.textAlign = labelRight ? "left" : "right";
        context.textBaseline = labelAbove ? "bottom" : "top";
        context.fillText(
          point.bits,
          x + (labelRight ? 7 : -7),
          y + (labelAbove ? -5 : 5),
        );
      });
    }

    pointerLocation(event) {
      const rect = this.canvas.getBoundingClientRect();
      return {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top,
      };
    }

    inPlot(point) {
      return point.x >= this.plot.left && point.x <= this.plot.right && point.y >= this.plot.top && point.y <= this.plot.bottom;
    }

    onWheel(event) {
      const point = this.pointerLocation(event);
      if (!this.inPlot(point)) return;
      event.preventDefault();
      const factor = Math.exp(clamp(event.deltaY, -180, 180) * 0.0024);
      const base = this.dataDomain();
      const xSpan = this.view.xMax - this.view.xMin;
      const ySpan = this.view.yMax - this.view.yMin;
      const newXSpan = clamp(xSpan * factor, (base.xMax - base.xMin) * MIN_VIEW_FRACTION, (base.xMax - base.xMin) * MAX_VIEW_MULTIPLIER);
      const newYSpan = clamp(ySpan * factor, (base.yMax - base.yMin) * MIN_VIEW_FRACTION, (base.yMax - base.yMin) * MAX_VIEW_MULTIPLIER);
      const xFraction = (point.x - this.plot.left) / (this.plot.right - this.plot.left);
      const yFraction = (this.plot.bottom - point.y) / (this.plot.bottom - this.plot.top);
      const xValue = this.pixelToX(point.x);
      const transformedY = this.view.yMin + yFraction * ySpan;
      const nextView = {
        xMin: xValue - xFraction * newXSpan,
        xMax: xValue + (1 - xFraction) * newXSpan,
        yMin: transformedY - yFraction * newYSpan,
        yMax: transformedY + (1 - yFraction) * newYSpan,
      };
      if (this.hasRightAxis()) {
        const rightSpan = this.view.rightYMax - this.view.rightYMin;
        const newRightSpan = clamp(
          rightSpan * factor,
          (base.rightYMax - base.rightYMin) * MIN_VIEW_FRACTION,
          (base.rightYMax - base.rightYMin) * MAX_VIEW_MULTIPLIER,
        );
        const transformedRightY = this.view.rightYMin + yFraction * rightSpan;
        nextView.rightYMin = transformedRightY - yFraction * newRightSpan;
        nextView.rightYMax = transformedRightY + (1 - yFraction) * newRightSpan;
      }
      this.view = nextView;
      this.hideTooltip();
      this.draw();
    }

    onPointerDown(event) {
      const point = this.pointerLocation(event);
      if (!this.inPlot(point)) return;
      this.drag = { point, view: { ...this.view }, moved: false };
      this.canvas.setPointerCapture(event.pointerId);
      this.canvas.classList.add("is-panning");
      this.hideTooltip();
    }

    onPointerMove(event) {
      const point = this.pointerLocation(event);
      if (this.drag) {
        const dx = point.x - this.drag.point.x;
        const dy = point.y - this.drag.point.y;
        this.drag.moved = this.drag.moved || Math.abs(dx) + Math.abs(dy) > 3;
        const xShift = -dx / (this.plot.right - this.plot.left) * (this.drag.view.xMax - this.drag.view.xMin);
        const yShift = dy / (this.plot.bottom - this.plot.top) * (this.drag.view.yMax - this.drag.view.yMin);
        const nextView = {
          xMin: this.drag.view.xMin + xShift,
          xMax: this.drag.view.xMax + xShift,
          yMin: this.drag.view.yMin + yShift,
          yMax: this.drag.view.yMax + yShift,
        };
        if (this.hasRightAxis()) {
          const rightShift = dy / (this.plot.bottom - this.plot.top)
            * (this.drag.view.rightYMax - this.drag.view.rightYMin);
          nextView.rightYMin = this.drag.view.rightYMin + rightShift;
          nextView.rightYMax = this.drag.view.rightYMax + rightShift;
        }
        this.view = nextView;
        this.draw();
        return;
      }
      if (!this.inPlot(point)) {
        this.hideTooltip();
        return;
      }
      this.showTooltip(point);
    }

    onPointerUp(event) {
      if (!this.drag) return;
      if (this.canvas.hasPointerCapture(event.pointerId)) this.canvas.releasePointerCapture(event.pointerId);
      this.drag = null;
      this.canvas.classList.remove("is-panning");
    }

    showTooltip(point) {
      if (this.spec.type === "decision-boundary") {
        this.tooltip.innerHTML = `<strong>Received I/Q</strong><br>I: ${escapeHtml(formatValue(this.pixelToX(point.x)))}<br>Q: ${escapeHtml(formatValue(this.pixelToY(point.y)))}`;
      } else {
        const visible = (this.spec.series || []).filter((series) => !this.hidden.has(series.id));
        const candidates = visible.flatMap((series) => (series.values || []).map((value) => ({ series, value })));
        if (!candidates.length) return;
        const nearest = candidates.reduce((best, item) => {
          const distance = Math.abs(this.xToPixel(item.value[0]) - point.x);
          return !best || distance < best.distance ? { ...item, distance } : best;
        }, null);
        const x = Number(nearest.value[0]);
        const lines = visible.map((series) => {
          const match = (series.values || []).find((value) => Number(value[0]) === x);
          if (!match) return "";
          const interval = (series.range || []).find(
            (value) => Number(value[0]) === x && Number(value[2]) > Number(value[1]),
          );
          const intervalText = interval
            ? ` <span class="noema-chart-tooltip-interval">[95% interval ${escapeHtml(formatValue(interval[1]))}–${escapeHtml(formatValue(interval[2]))}; n=${escapeHtml(series.sampleCount || "—")}]</span>`
            : "";
          return `<span class="noema-chart-tooltip-swatch" style="background:${escapeHtml(series.color)}"></span>${escapeHtml(series.label)}: <strong>${escapeHtml(formatValue(match[1]))}</strong>${intervalText}`;
        }).filter(Boolean);
        this.tooltip.innerHTML = `<strong>${escapeHtml(this.spec.xLabel)}: ${escapeHtml(formatValue(x))}</strong><br>${lines.join("<br>")}`;
      }
      this.tooltip.hidden = false;
      const stage = this.root.querySelector(".noema-chart-stage");
      const maxLeft = Math.max(8, stage.clientWidth - this.tooltip.offsetWidth - 8);
      const maxTop = Math.max(8, stage.clientHeight - this.tooltip.offsetHeight - 8);
      this.tooltip.style.left = `${clamp(point.x + 14, 8, maxLeft)}px`;
      this.tooltip.style.top = `${clamp(point.y + 14, 8, maxTop)}px`;
    }

    hideTooltip() {
      if (this.tooltip) this.tooltip.hidden = true;
    }
  }

  function enhanceAccessibleTables() {
    document.querySelectorAll("table.noema-compact-table").forEach((table) => {
      table.querySelectorAll("thead th").forEach((header) => {
        header.setAttribute("scope", "col");
      });
      table.querySelectorAll("tbody tr").forEach((row) => {
        const cell = row.querySelector(":scope > td:first-child");
        if (!cell) return;
        const header = document.createElement("th");
        Array.from(cell.attributes).forEach((attribute) => {
          header.setAttribute(attribute.name, attribute.value);
        });
        header.setAttribute("scope", "row");
        header.innerHTML = cell.innerHTML;
        cell.replaceWith(header);
      });
      const wrapper = table.closest(".pst-scrollable-table-container");
      const caption = table.querySelector("caption");
      if (wrapper && caption) {
        wrapper.setAttribute("role", "region");
        wrapper.setAttribute(
          "aria-label",
          `${caption.textContent.replace("#", "").trim()} table; scroll horizontally for additional columns`,
        );
      }
    });
  }

  function legacyCopy(text) {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    const copied = document.execCommand("copy");
    textarea.remove();
    if (!copied) throw new Error("copy command was rejected");
  }

  function enhanceCodeBlocks() {
    const selector = [
      "div.highlight-bash",
      "div.highlight-shell",
      "div.highlight-console",
      "div.highlight-text",
      "div.highlight-yaml",
    ].join(",");
    document.querySelectorAll(selector).forEach((block) => {
      if (block.dataset.noemaCopyReady === "true") return;
      const pre = block.querySelector("pre");
      if (!pre) return;
      block.dataset.noemaCopyReady = "true";
      block.classList.add("noema-copyable-code");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "noema-copy-button";
      button.textContent = "Copy";
      button.setAttribute("aria-label", "Copy this code block to the clipboard");
      button.addEventListener("click", async () => {
        try {
          const text = pre.textContent || "";
          if (navigator.clipboard && window.isSecureContext) {
            await navigator.clipboard.writeText(text);
          } else {
            legacyCopy(text);
          }
          button.textContent = "Copied";
          button.setAttribute("aria-label", "Code copied to the clipboard");
        } catch (error) {
          button.textContent = "Copy failed";
          button.setAttribute("aria-label", "Copy failed; select the code manually");
        }
        window.setTimeout(() => {
          button.textContent = "Copy";
          button.setAttribute("aria-label", "Copy this code block to the clipboard");
        }, 1800);
      });
      block.appendChild(button);
    });
  }

  function initializeCharts() {
    const specs = window.NOEMA_DEMO_CHARTS || {};
    document.querySelectorAll(CHART_SELECTOR).forEach((root) => {
      if (root.dataset.noemaChartReady === "true") return;
      const id = root.dataset.noemaChart;
      const spec = specs[id];
      if (!spec) {
        root.textContent = `Interactive chart data is unavailable: ${id}`;
        root.classList.add("noema-doc-chart-error");
        return;
      }
      root.dataset.noemaChartReady = "true";
      root.noemaChart = new NoemaDemoChart(root, spec);
    });
  }

  window.NOEMA_DEMO_CHART_RUNTIME = Object.freeze({
    initialize: initializeCharts,
    redraw() {
      document.querySelectorAll(CHART_SELECTOR).forEach((root) => {
        if (root.noemaChart) root.noemaChart.draw();
      });
    },
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
      enhanceAccessibleTables();
      enhanceCodeBlocks();
      initializeCharts();
    }, { once: true });
  } else {
    enhanceAccessibleTables();
    enhanceCodeBlocks();
    initializeCharts();
  }

  new MutationObserver(() => {
    document.querySelectorAll(CHART_SELECTOR).forEach((root) => {
      if (root.noemaChart) root.noemaChart.draw();
    });
  }).observe(document.documentElement, { attributes: true, attributeFilter: ["class", "data-theme"] });
})();
