(function () {
  "use strict";

  const THEME_STORAGE_KEY = "noema.theme";
  const DEFAULT_FACET = "overview";
  const DEFAULT_PRIMARY_VIEW = "graph";
  const DEFAULT_EXPERIMENT_ID = "learned_qpsk_demapper_demo";
  const GRAPH_NODE_WIDTH = 210;
  const GRAPH_NODE_HEIGHT = 96;
  const GRAPH_COLUMN_GAP = 54;
  const GRAPH_ROW_GAP = 38;
  const GRAPH_MARGIN = 48;

  const state = {
    catalog: null,
    experiment: null,
    facet: DEFAULT_FACET,
    primaryView: DEFAULT_PRIMARY_VIEW,
    graphZoom: 1,
    graphLayout: null,
    selectedGraphNodeId: null,
    loadedScripts: new Set(),
    selectedTables: new Map(),
  };

  const elements = {
    experimentMeta: document.getElementById("experimentMeta"),
    experimentSelect: document.getElementById("experimentSelect"),
    experimentTitle: document.getElementById("experimentTitle"),
    graphTabButton: document.getElementById("graphTabButton"),
    graphView: document.getElementById("graphView"),
    recipeGraph: document.getElementById("recipeGraph"),
    resultsComparison: document.getElementById("resultsComparison"),
    resultsPanel: document.getElementById("resultsPanel"),
    resultsTabButton: document.getElementById("resultsTabButton"),
    resultsView: document.getElementById("resultsView"),
    sourceStatus: document.getElementById("sourceStatus"),
    themeToggleButton: document.getElementById("themeToggleButton"),
    toast: document.getElementById("toast"),
  };

  initializeTheme();
  bindShell();
  loadCatalog();

  function initializeTheme() {
    let theme = "dark";
    try {
      theme = localStorage.getItem(THEME_STORAGE_KEY) === "light" ? "light" : "dark";
    } catch {
      // Storage is optional; the static result explorer still works without it.
    }
    setTheme(theme, false);
  }

  function bindShell() {
    elements.themeToggleButton.addEventListener("click", () => {
      const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
      setTheme(next, true);
    });
    elements.experimentSelect.addEventListener("change", () => {
      selectExperiment(elements.experimentSelect.value, true);
    });
    [elements.graphTabButton, elements.resultsTabButton].forEach((button, index, buttons) => {
      button.addEventListener("click", () => setPrimaryView(button.dataset.hostedView, true));
      button.addEventListener("keydown", (event) => {
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
        event.preventDefault();
        let nextIndex = index;
        if (event.key === "ArrowLeft") nextIndex = (index - 1 + buttons.length) % buttons.length;
        if (event.key === "ArrowRight") nextIndex = (index + 1) % buttons.length;
        if (event.key === "Home") nextIndex = 0;
        if (event.key === "End") nextIndex = buttons.length - 1;
        setPrimaryView(buttons[nextIndex].dataset.hostedView, true);
        buttons[nextIndex].focus();
      });
    });
  }

  function setTheme(theme, persist) {
    const next = theme === "light" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    const label = next === "dark" ? "Switch to light mode" : "Switch to dark mode";
    elements.themeToggleButton.setAttribute("aria-label", label);
    elements.themeToggleButton.setAttribute("title", label);
    if (persist) {
      try {
        localStorage.setItem(THEME_STORAGE_KEY, next);
      } catch {
        // Theme persistence is a convenience.
      }
    }
    window.requestAnimationFrame(() => {
      window.NOEMA_DEMO_CHART_RUNTIME?.redraw();
    });
  }

  async function loadCatalog() {
    try {
      let catalog = window.NOEMA_HOSTED_DEMO_CATALOG;
      if (!catalog) {
        const response = await fetch("catalog.json");
        if (!response.ok) {
          throw new Error(`catalog request returned ${response.status}`);
        }
        catalog = await response.json();
      }
      if (
        catalog.kind !== "noema.hosted_demo_catalog"
        || !Array.isArray(catalog.experiments)
        || catalog.experiments.length === 0
      ) {
        throw new Error("catalog has no documentation experiments");
      }
      state.catalog = catalog;
      populateExperimentPicker(catalog.experiments);
      const parameters = new URLSearchParams(window.location.search);
      const requested = parameters.get("experiment");
      const requestedFacet = parameters.get("view");
      const requestedPrimaryView = parameters.get("tab");
      const initial = catalog.experiments.some((item) => item.id === requested)
        ? requested
        : catalog.experiments.some((item) => item.id === DEFAULT_EXPERIMENT_ID)
          ? DEFAULT_EXPERIMENT_ID
          : catalog.experiments[0].id;
      await selectExperiment(initial, false);
      if (requestedFacet && elements.resultsComparison.querySelector(
        `[data-result-facet="${cssEscape(requestedFacet)}"]`,
      )) {
        activateFacet(requestedFacet, false);
        setPrimaryView("results", false);
      } else if (requestedPrimaryView === "results") {
        setPrimaryView("results", false);
      } else {
        setPrimaryView(DEFAULT_PRIMARY_VIEW, false);
      }
    } catch (error) {
      renderFatalError(`Could not load the hosted demo catalog: ${error.message}`);
    } finally {
      elements.resultsPanel.setAttribute("aria-busy", "false");
    }
  }

  function populateExperimentPicker(experiments) {
    elements.experimentSelect.innerHTML = experiments.map((experiment) => (
      `<option value="${escapeAttr(experiment.id)}">${escapeHtml(experiment.title)}</option>`
    )).join("");
    elements.experimentSelect.disabled = false;
  }

  async function selectExperiment(id, updateLocation) {
    const experiment = state.catalog.experiments.find((item) => item.id === id);
    if (!experiment) return;

    state.experiment = experiment;
    state.facet = DEFAULT_FACET;
    state.selectedGraphNodeId = null;
    elements.experimentSelect.value = experiment.id;
    elements.experimentTitle.textContent = experiment.title;
    elements.experimentMeta.textContent = experiment.description;
    elements.sourceStatus.textContent = "documentation snapshot";
    elements.resultsPanel.setAttribute("aria-busy", "true");
    elements.resultsComparison.innerHTML = (
      '<div class="result-empty hosted-loading">Loading the shared result presentation…</div>'
    );
    elements.recipeGraph.innerHTML = (
      '<div class="result-empty hosted-loading">Loading the shared recipe graph…</div>'
    );

    if (updateLocation) {
      const url = new URL(window.location.href);
      url.searchParams.set("experiment", experiment.id);
      url.searchParams.delete("view");
      if (state.primaryView === DEFAULT_PRIMARY_VIEW) url.searchParams.delete("tab");
      else url.searchParams.set("tab", state.primaryView);
      window.history.replaceState({}, "", url);
    }

    try {
      await ensureChartScripts(experiment.chart_scripts || []);
      renderExperiment(experiment);
      renderRecipeGraph(experiment.recipe);
      window.NOEMA_DEMO_CHART_RUNTIME?.initialize();
    } catch (error) {
      renderFatalError(`Could not render ${experiment.title}: ${error.message}`);
    } finally {
      elements.resultsPanel.setAttribute("aria-busy", "false");
    }
  }

  function setPrimaryView(view, updateLocation) {
    const next = view === "results" ? "results" : DEFAULT_PRIMARY_VIEW;
    state.primaryView = next;
    [elements.graphTabButton, elements.resultsTabButton].forEach((button) => {
      const active = button.dataset.hostedView === next;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", String(active));
      button.tabIndex = active ? 0 : -1;
    });
    elements.graphView.hidden = next !== "graph";
    elements.resultsView.hidden = next !== "results";
    elements.sourceStatus.textContent = next === "graph"
      ? "checked-in recipe"
      : "documentation snapshot";

    if (updateLocation) {
      const url = new URL(window.location.href);
      if (next === DEFAULT_PRIMARY_VIEW) {
        url.searchParams.delete("tab");
        url.searchParams.delete("view");
      } else {
        url.searchParams.set("tab", next);
        if (state.facet === DEFAULT_FACET) url.searchParams.delete("view");
        else url.searchParams.set("view", state.facet);
      }
      window.history.replaceState({}, "", url);
    }

    window.requestAnimationFrame(() => {
      if (next === "graph") applyGraphZoom();
      else window.NOEMA_DEMO_CHART_RUNTIME?.redraw();
    });
  }

  async function ensureChartScripts(scripts) {
    for (const source of scripts) {
      if (state.loadedScripts.has(source)) continue;
      await loadScript(source);
      state.loadedScripts.add(source);
    }
  }

  function loadScript(source) {
    return new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = source;
      script.async = false;
      script.addEventListener("load", resolve, { once: true });
      script.addEventListener(
        "error",
        () => reject(new Error(`could not load ${source}`)),
        { once: true },
      );
      document.head.appendChild(script);
    });
  }

  function renderExperiment(experiment) {
    disconnectRenderedCharts();
    const facets = [
      { id: "overview", label: "Overview" },
      ...(experiment.tables.length ? [{ id: "data", label: "Result data" }] : []),
      { id: "evidence", label: "Evidence" },
    ];
    elements.resultsComparison.innerHTML = `
      ${renderFacetTabs(facets)}
      <div class="results-facet-panels">
        ${renderFacetPanel("overview", renderOverview(experiment), true)}
        ${experiment.tables.length ? renderFacetPanel("data", renderDataFacet(experiment), false) : ""}
        ${renderFacetPanel("evidence", renderEvidence(experiment), false)}
      </div>
    `;
    bindFacetTabs();
    bindTableControls(experiment);
  }

  function renderRecipeGraph(recipe) {
    if (!recipe || !Array.isArray(recipe.steps) || !recipe.steps.length) {
      elements.recipeGraph.innerHTML = (
        '<div class="hosted-error">This demo does not declare a representative recipe.</div>'
      );
      return;
    }

    const layout = buildRecipeLayout(recipe.steps);
    state.graphLayout = layout;
    state.graphZoom = 1;
    elements.recipeGraph.innerHTML = `
      <div class="hosted-graph-shell">
        <section class="hosted-graph-stage" aria-label="Read-only recipe graph">
          <div class="hosted-graph-toolbar">
            <div class="hosted-graph-provenance">
              <span class="hosted-graph-readonly">read only</span>
              <strong>${escapeHtml(recipe.name)}</strong>
              <span>${layout.nodes.length} blocks · ${layout.edges.length} connections</span>
            </div>
            <div class="hosted-graph-zoom" aria-label="Graph zoom controls">
              <button type="button" class="compact-button" data-graph-zoom-action="out" aria-label="Zoom out">−</button>
              <span class="hosted-graph-zoom-value">100%</span>
              <button type="button" class="compact-button" data-graph-zoom-action="in" aria-label="Zoom in">+</button>
              <button type="button" class="compact-button" data-graph-zoom-action="fit">Fit</button>
            </div>
          </div>
          <div class="graph-surface hosted-graph-surface" id="hostedGraphSurface">
            ${renderRecipeSvg(layout)}
          </div>
        </section>
        <aside class="hosted-recipe-inspector" aria-label="Recipe details">
          <div class="panel-header">
            <strong>Recipe details</strong>
            <span class="suite-status-tag ${escapeAttr(recipe.suite?.status || "unknown")}">
              ${escapeHtml(recipe.suite?.status || "recipe")}
            </span>
          </div>
          <div id="hostedRecipeInspectorBody" class="hosted-recipe-inspector-body">
            ${renderRecipeInspector(recipe, null)}
          </div>
        </aside>
      </div>
    `;
    bindRecipeGraph(recipe);
    window.requestAnimationFrame(fitRecipeGraph);
  }

  function buildRecipeLayout(steps) {
    const nodes = steps.map((step, order) => ({
      ...step,
      order,
      family: String(step.op || "block").split(".")[0].toLowerCase(),
    }));
    const nodeById = new Map(nodes.map((node) => [String(node.id), node]));
    const edges = [];
    nodes.forEach((node) => {
      Object.entries(node.inputs || {}).forEach(([input, reference], inputIndex) => {
        if (typeof reference !== "string") return;
        const separator = reference.indexOf(".");
        if (separator < 1) return;
        const from = reference.slice(0, separator);
        if (!nodeById.has(from)) return;
        edges.push({
          id: `${from}:${reference.slice(separator + 1)}:${node.id}:${input}`,
          from,
          fromOutput: reference.slice(separator + 1),
          to: String(node.id),
          toInput: input,
          inputIndex,
        });
      });
    });

    const indegree = new Map(nodes.map((node) => [String(node.id), 0]));
    const outgoing = new Map(nodes.map((node) => [String(node.id), []]));
    edges.forEach((edge) => {
      indegree.set(edge.to, (indegree.get(edge.to) || 0) + 1);
      outgoing.get(edge.from)?.push(edge);
    });

    const rank = new Map(nodes.map((node) => [String(node.id), 0]));
    const queue = nodes
      .filter((node) => indegree.get(String(node.id)) === 0)
      .sort((left, right) => left.order - right.order);
    const visited = new Set();
    while (queue.length) {
      const node = queue.shift();
      const nodeId = String(node.id);
      visited.add(nodeId);
      (outgoing.get(nodeId) || []).forEach((edge) => {
        rank.set(edge.to, Math.max(rank.get(edge.to) || 0, (rank.get(nodeId) || 0) + 1));
        indegree.set(edge.to, (indegree.get(edge.to) || 0) - 1);
        if (indegree.get(edge.to) === 0) {
          queue.push(nodeById.get(edge.to));
          queue.sort((left, right) => left.order - right.order);
        }
      });
    }

    nodes.filter((node) => !visited.has(String(node.id))).forEach((node) => {
      const parentRanks = edges
        .filter((edge) => edge.to === String(node.id))
        .map((edge) => rank.get(edge.from) || 0);
      rank.set(String(node.id), parentRanks.length ? Math.max(...parentRanks) + 1 : 0);
    });

    const columns = new Map();
    nodes.forEach((node) => {
      const nodeRank = rank.get(String(node.id)) || 0;
      if (!columns.has(nodeRank)) columns.set(nodeRank, []);
      columns.get(nodeRank).push(node);
    });
    const maxRank = Math.max(0, ...columns.keys());
    const largestColumn = Math.max(1, ...Array.from(columns.values(), (items) => items.length));
    const innerHeight = largestColumn * GRAPH_NODE_HEIGHT
      + Math.max(0, largestColumn - 1) * GRAPH_ROW_GAP;
    const height = Math.max(430, innerHeight + GRAPH_MARGIN * 2);
    const width = Math.max(
      760,
      GRAPH_MARGIN * 2
        + (maxRank + 1) * GRAPH_NODE_WIDTH
        + maxRank * GRAPH_COLUMN_GAP,
    );
    const positions = new Map();
    columns.forEach((columnNodes, columnRank) => {
      const columnHeight = columnNodes.length * GRAPH_NODE_HEIGHT
        + Math.max(0, columnNodes.length - 1) * GRAPH_ROW_GAP;
      const top = (height - columnHeight) / 2;
      columnNodes
        .sort((left, right) => left.order - right.order)
        .forEach((node, index) => {
          positions.set(String(node.id), {
            x: GRAPH_MARGIN + columnRank * (GRAPH_NODE_WIDTH + GRAPH_COLUMN_GAP),
            y: top + index * (GRAPH_NODE_HEIGHT + GRAPH_ROW_GAP),
          });
        });
    });
    return { nodes, edges, positions, width, height };
  }

  function renderRecipeSvg(layout) {
    const incoming = new Map(layout.nodes.map((node) => [String(node.id), []]));
    const outgoing = new Map(layout.nodes.map((node) => [String(node.id), []]));
    layout.edges.forEach((edge) => {
      incoming.get(edge.to)?.push(edge);
      outgoing.get(edge.from)?.push(edge);
    });

    const edgeMarkup = layout.edges.map((edge) => {
      const from = layout.positions.get(edge.from);
      const to = layout.positions.get(edge.to);
      const sourceEdges = outgoing.get(edge.from) || [];
      const targetEdges = incoming.get(edge.to) || [];
      const sourceIndex = sourceEdges.indexOf(edge);
      const targetIndex = targetEdges.indexOf(edge);
      const x1 = from.x + GRAPH_NODE_WIDTH;
      const y1 = graphPortY(from.y, sourceIndex, sourceEdges.length);
      const x2 = to.x;
      const y2 = graphPortY(to.y, targetIndex, targetEdges.length);
      const control = Math.max(24, (x2 - x1) * 0.44);
      const path = `M ${x1} ${y1} C ${x1 + control} ${y1}, ${x2 - control} ${y2}, ${x2} ${y2}`;
      const label = truncateText(`${edge.fromOutput} → ${edge.toInput}`, 27);
      return `
        <g class="hosted-recipe-edge">
          <path class="edge" d="${path}" marker-end="url(#hostedGraphArrow)"></path>
          <text class="edge-label" x="${(x1 + x2) / 2}" y="${(y1 + y2) / 2 - 7}">
            ${escapeHtml(label)}
          </text>
        </g>
      `;
    }).join("");

    const nodeMarkup = layout.nodes.map((node) => {
      const position = layout.positions.get(String(node.id));
      const inputs = incoming.get(String(node.id)) || [];
      const outputs = outgoing.get(String(node.id)) || [];
      const flow = `${inputs.length} in · ${outputs.length} out`;
      return `
        <g
          class="node hosted-recipe-node ${escapeAttr(node.family)}"
          transform="translate(${position.x} ${position.y})"
          data-graph-node="${escapeAttr(node.id)}"
          role="button"
          tabindex="0"
          aria-label="${escapeAttr(`${humanizeId(node.id)} block, ${node.op}`)}"
        >
          <title>${escapeHtml(`${node.id}: ${node.op}`)}</title>
          <rect width="${GRAPH_NODE_WIDTH}" height="${GRAPH_NODE_HEIGHT}"></rect>
          <text class="node-title" x="16" y="24">${escapeHtml(truncateText(humanizeId(node.id), 27))}</text>
          <text class="hosted-node-family" x="${GRAPH_NODE_WIDTH - 14}" y="24" text-anchor="end">
            ${escapeHtml(node.family)}
          </text>
          <text class="node-detail" x="16" y="49">${escapeHtml(truncateText(node.op, 30))}</text>
          <text class="node-meta" x="16" y="75">${escapeHtml(flow)}</text>
          ${inputs.map((edge, index) => `
            <circle
              class="node-port node-input-port connected"
              cx="0"
              cy="${graphPortY(0, index, inputs.length)}"
              r="5"
            ></circle>
          `).join("")}
          ${outputs.map((edge, index) => `
            <circle
              class="node-port connected"
              cx="${GRAPH_NODE_WIDTH}"
              cy="${graphPortY(0, index, outputs.length)}"
              r="5"
            ></circle>
          `).join("")}
        </g>
      `;
    }).join("");

    return `
      <svg
        id="hostedRecipeSvg"
        class="hosted-recipe-svg"
        viewBox="0 0 ${layout.width} ${layout.height}"
        width="${layout.width}"
        height="${layout.height}"
        role="img"
        aria-label="Recipe pipeline with ${layout.nodes.length} blocks and ${layout.edges.length} connections"
      >
        <defs>
          <marker id="hostedGraphArrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M 0 0 L 8 4 L 0 8 z" class="hosted-edge-arrow"></path>
          </marker>
        </defs>
        ${edgeMarkup}
        ${nodeMarkup}
      </svg>
    `;
  }

  function graphPortY(nodeTop, index, count) {
    if (!count) return nodeTop + GRAPH_NODE_HEIGHT / 2;
    const available = GRAPH_NODE_HEIGHT - 36;
    return nodeTop + 18 + ((index + 1) * available) / (count + 1);
  }

  function bindRecipeGraph(recipe) {
    elements.recipeGraph.querySelectorAll("[data-graph-node]").forEach((node) => {
      node.addEventListener("click", () => selectGraphNode(recipe, node.dataset.graphNode));
      node.addEventListener("keydown", (event) => {
        if (!["Enter", " "].includes(event.key)) return;
        event.preventDefault();
        selectGraphNode(recipe, node.dataset.graphNode);
      });
    });
    elements.recipeGraph.querySelectorAll("[data-graph-zoom-action]").forEach((button) => {
      button.addEventListener("click", () => {
        const action = button.dataset.graphZoomAction;
        if (action === "fit") {
          fitRecipeGraph();
          return;
        }
        state.graphZoom = clampGraphZoom(
          state.graphZoom + (action === "in" ? 0.15 : -0.15),
        );
        applyGraphZoom();
      });
    });
  }

  function selectGraphNode(recipe, nodeId) {
    state.selectedGraphNodeId = nodeId;
    elements.recipeGraph.querySelectorAll("[data-graph-node]").forEach((node) => {
      node.classList.toggle("selected", node.dataset.graphNode === nodeId);
    });
    const body = document.getElementById("hostedRecipeInspectorBody");
    const step = recipe.steps.find((item) => String(item.id) === nodeId) || null;
    if (body) body.innerHTML = renderRecipeInspector(recipe, step);
  }

  function renderRecipeInspector(recipe, step) {
    if (!step) {
      const profile = recipe.execution_profile?.id || "not declared";
      return `
        <section class="inspector-section hosted-recipe-summary">
          <span class="hosted-inspector-eyebrow">Representative demo recipe</span>
          <h2>${escapeHtml(recipe.name)}</h2>
          <p>${escapeHtml(recipe.description || "No recipe description is declared.")}</p>
        </section>
        <section class="inspector-section hosted-recipe-facts">
          ${inspectorFact("Source", recipe.source_path)}
          ${inspectorFact("Execution profile", profile)}
          ${inspectorFact("Suite", recipe.suite?.name || recipe.suite?.id || "not declared")}
          ${inspectorFact("Blocks", recipe.steps.length)}
        </section>
        <section class="inspector-section hosted-inspector-help">
          Select a block to inspect its exact operation, inputs, and parameters.
        </section>
      `;
    }

    const inputs = Object.entries(step.inputs || {});
    const params = Object.entries(step.params || {});
    return `
      <section class="inspector-section hosted-recipe-summary">
        <span class="hosted-inspector-eyebrow">Selected block</span>
        <h2>${escapeHtml(humanizeId(step.id))}</h2>
        <code>${escapeHtml(step.id)}</code>
      </section>
      <section class="inspector-section hosted-recipe-facts">
        ${inspectorFact("Operation", step.op)}
      </section>
      <section class="inspector-section">
        <h3>Input connections</h3>
        ${inputs.length ? `
          <dl class="hosted-inspector-list">
            ${inputs.map(([name, reference]) => `
              <div>
                <dt>${escapeHtml(name)}</dt>
                <dd title="${escapeAttr(formatGraphValue(reference))}">${escapeHtml(formatGraphValue(reference))}</dd>
              </div>
            `).join("")}
          </dl>
        ` : '<p class="hosted-inspector-empty">This source block has no inputs.</p>'}
      </section>
      <section class="inspector-section">
        <h3>Parameters</h3>
        ${params.length ? `
          <dl class="hosted-inspector-list">
            ${params.map(([name, value]) => {
              const formatted = formatGraphValue(value);
              return `
                <div>
                  <dt>${escapeHtml(name)}</dt>
                  <dd title="${escapeAttr(formatted)}">${escapeHtml(truncateText(formatted, 88))}</dd>
                </div>
              `;
            }).join("")}
          </dl>
        ` : '<p class="hosted-inspector-empty">This block uses its declared defaults.</p>'}
      </section>
    `;
  }

  function inspectorFact(label, value) {
    return `
      <div class="hosted-inspector-fact">
        <span>${escapeHtml(label)}</span>
        <strong title="${escapeAttr(value)}">${escapeHtml(value)}</strong>
      </div>
    `;
  }

  function fitRecipeGraph() {
    const surface = document.getElementById("hostedGraphSurface");
    if (!surface || !state.graphLayout || surface.clientWidth === 0) return;
    const widthScale = (surface.clientWidth - 28) / state.graphLayout.width;
    const heightScale = (surface.clientHeight - 28) / state.graphLayout.height;
    state.graphZoom = clampGraphZoom(Math.min(1, widthScale, heightScale));
    applyGraphZoom();
    surface.scrollTo({ left: 0, top: 0 });
  }

  function applyGraphZoom() {
    const svg = document.getElementById("hostedRecipeSvg");
    if (!svg || !state.graphLayout) return;
    svg.style.width = `${Math.round(state.graphLayout.width * state.graphZoom)}px`;
    svg.style.height = `${Math.round(state.graphLayout.height * state.graphZoom)}px`;
    const label = elements.recipeGraph.querySelector(".hosted-graph-zoom-value");
    if (label) label.textContent = `${Math.round(state.graphZoom * 100)}%`;
  }

  function clampGraphZoom(value) {
    return Math.min(1.6, Math.max(0.15, value));
  }

  function humanizeId(value) {
    return String(value || "block")
      .replace(/[._-]+/g, " ")
      .replace(/\b\w/g, (character) => character.toUpperCase());
  }

  function formatGraphValue(value) {
    if (typeof value === "string") return value;
    if (value === null || value === undefined) return String(value ?? "");
    if (typeof value === "object") return JSON.stringify(value);
    return String(value);
  }

  function truncateText(value, length) {
    const text = String(value ?? "");
    return text.length > length ? `${text.slice(0, Math.max(1, length - 1))}…` : text;
  }

  function disconnectRenderedCharts() {
    elements.resultsComparison.querySelectorAll("[data-noema-chart]").forEach((root) => {
      if (root.noemaChart?.resizeObserver) {
        root.noemaChart.resizeObserver.disconnect();
      }
    });
  }

  function renderFacetTabs(facets) {
    return `
      <nav class="results-facet-tabs" role="tablist" aria-label="Result sections">
        ${facets.map((facet, index) => `
          <button
            type="button"
            class="results-facet-button${index === 0 ? " active" : ""}"
            id="hosted-facet-tab-${escapeAttr(facet.id)}"
            role="tab"
            aria-selected="${index === 0 ? "true" : "false"}"
            aria-controls="hosted-facet-panel-${escapeAttr(facet.id)}"
            tabindex="${index === 0 ? "0" : "-1"}"
            data-result-facet="${escapeAttr(facet.id)}"
          >${escapeHtml(facet.label)}</button>
        `).join("")}
      </nav>
    `;
  }

  function renderFacetPanel(id, markup, active) {
    return `
      <section
        id="hosted-facet-panel-${escapeAttr(id)}"
        class="results-facet-panel"
        role="tabpanel"
        aria-labelledby="hosted-facet-tab-${escapeAttr(id)}"
        data-result-facet-panel="${escapeAttr(id)}"
        ${active ? "" : "hidden"}
      >${markup}</section>
    `;
  }

  function bindFacetTabs() {
    const buttons = Array.from(
      elements.resultsComparison.querySelectorAll("[data-result-facet]"),
    );
    buttons.forEach((button, index) => {
      button.addEventListener("click", () => activateFacet(button.dataset.resultFacet, true));
      button.addEventListener("keydown", (event) => {
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
        event.preventDefault();
        let nextIndex = index;
        if (event.key === "ArrowLeft") nextIndex = (index - 1 + buttons.length) % buttons.length;
        if (event.key === "ArrowRight") nextIndex = (index + 1) % buttons.length;
        if (event.key === "Home") nextIndex = 0;
        if (event.key === "End") nextIndex = buttons.length - 1;
        activateFacet(buttons[nextIndex].dataset.resultFacet, true);
        buttons[nextIndex].focus();
      });
    });
  }

  function activateFacet(id, updateLocation) {
    state.facet = id;
    elements.resultsComparison.querySelectorAll("[data-result-facet]").forEach((button) => {
      const active = button.dataset.resultFacet === id;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", String(active));
      button.tabIndex = active ? 0 : -1;
    });
    elements.resultsComparison.querySelectorAll("[data-result-facet-panel]").forEach((panel) => {
      panel.hidden = panel.dataset.resultFacetPanel !== id;
    });
    if (id === "data") {
      loadSelectedTable(state.experiment);
    }
    if (updateLocation) {
      const url = new URL(window.location.href);
      if (id === DEFAULT_FACET) url.searchParams.delete("view");
      else url.searchParams.set("view", id);
      window.history.replaceState({}, "", url);
    }
    window.requestAnimationFrame(() => window.NOEMA_DEMO_CHART_RUNTIME?.redraw());
  }

  function renderOverview(experiment) {
    const snapshot = experiment.snapshot || {};
    const sourceHash = snapshot.manifest_sha256
      ? `${snapshot.manifest_sha256.slice(0, 12)}…`
      : "not declared";
    return `
      <section class="results-summary-strip">
        ${summaryPill("Charts", experiment.chart_ids.length)}
        ${summaryPill("Stored runs", finiteOrDash(snapshot.run_count))}
        ${summaryPill("Projected rows", finiteOrDash(snapshot.projection_rows))}
        ${summaryPill("Result tables", experiment.tables.length)}
        ${summaryPill("Evidence files", experiment.evidence.length)}
        ${(experiment.published_results || []).length
          ? summaryPill("Published pages", experiment.published_results.length)
          : ""}
        ${summaryPill("Manifest", sourceHash)}
      </section>
      <section class="results-overview hosted-chart-grid">
        ${experiment.chart_ids.map((chartId) => `
          <div class="hosted-chart-frame">
            <div data-noema-chart="${escapeAttr(chartId)}"></div>
          </div>
        `).join("")}
      </section>
    `;
  }

  function summaryPill(label, value) {
    return `
      <div class="results-summary-pill">
        <span>${escapeHtml(label)}</span>
        <strong title="${escapeAttr(value)}">${escapeHtml(value)}</strong>
      </div>
    `;
  }

  function renderDataFacet(experiment) {
    const selected = selectedTable(experiment);
    return `
      <section class="hosted-table-section">
        <div class="hosted-table-toolbar">
          <div>
            <strong>Stored result table</strong>
            <small id="tableEvidencePath">${escapeHtml(selected.path)}</small>
          </div>
          <label>
            <span>Table</span>
            <select id="resultTableSelect">
              ${experiment.tables.map((table) => `
                <option value="${escapeAttr(table.path)}"${table.path === selected.path ? " selected" : ""}>
                  ${escapeHtml(table.label)}
                </option>
              `).join("")}
            </select>
          </label>
        </div>
        <div id="resultTable" class="table-wrap table-wrap-tall hosted-table-wrap">
          <div class="result-empty">Open this tab to load the stored CSV.</div>
        </div>
      </section>
    `;
  }

  function selectedTable(experiment) {
    const selectedPath = state.selectedTables.get(experiment.id);
    return experiment.tables.find((table) => table.path === selectedPath) || experiment.tables[0];
  }

  function bindTableControls(experiment) {
    const select = document.getElementById("resultTableSelect");
    if (!select) return;
    select.addEventListener("change", () => {
      state.selectedTables.set(experiment.id, select.value);
      loadSelectedTable(experiment);
    });
  }

  async function loadSelectedTable(experiment) {
    const target = document.getElementById("resultTable");
    const pathLabel = document.getElementById("tableEvidencePath");
    if (!target || !pathLabel) return;
    const table = selectedTable(experiment);
    pathLabel.textContent = table.path;
    target.innerHTML = '<div class="result-empty">Loading stored CSV…</div>';
    try {
      let source = window.NOEMA_HOSTED_DEMO_TABLES?.[table.path];
      if (typeof source !== "string") {
        const response = await fetch(table.path);
        if (!response.ok) throw new Error(`request returned ${response.status}`);
        source = await response.text();
      }
      const rows = parseCsv(source);
      target.innerHTML = renderCsvTable(rows);
    } catch (error) {
      target.innerHTML = (
        `<div class="result-empty source-error">Could not load ${escapeHtml(table.path)}: ${escapeHtml(error.message)}</div>`
      );
    }
  }

  function parseCsv(source) {
    const rows = [];
    let row = [];
    let value = "";
    let quoted = false;
    for (let index = 0; index < source.length; index += 1) {
      const character = source[index];
      if (quoted) {
        if (character === '"' && source[index + 1] === '"') {
          value += '"';
          index += 1;
        } else if (character === '"') {
          quoted = false;
        } else {
          value += character;
        }
      } else if (character === '"') {
        quoted = true;
      } else if (character === ",") {
        row.push(value);
        value = "";
      } else if (character === "\n") {
        row.push(value.replace(/\r$/, ""));
        rows.push(row);
        row = [];
        value = "";
      } else {
        value += character;
      }
    }
    if (value.length || row.length) {
      row.push(value.replace(/\r$/, ""));
      rows.push(row);
    }
    return rows.filter((item) => item.some((cell) => cell !== ""));
  }

  function renderCsvTable(rows) {
    if (!rows.length) return '<div class="result-empty">The stored CSV is empty.</div>';
    const [header, ...body] = rows;
    return `
      <table>
        <thead>
          <tr>${header.map((cell) => `<th scope="col">${escapeHtml(cell)}</th>`).join("")}</tr>
        </thead>
        <tbody>
          ${body.map((row) => `
            <tr>
              ${header.map((_, index) => `<td>${escapeHtml(row[index] ?? "")}</td>`).join("")}
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
  }

  function renderEvidence(experiment) {
    const snapshot = experiment.snapshot || {};
    return `
      <section class="hosted-evidence-section">
        <div class="hosted-evidence-intro">
          <strong>Traceable static evidence.</strong>
          This experiment was discovered from <code>${escapeHtml(experiment.source_path)}</code>.
          Its chart IDs, downloadable tables, manifests, hashes, and run counts come from the same
          checked-in assets used to render that documentation page.
          ${snapshot.kind ? `<br>Snapshot kind: <code>${escapeHtml(snapshot.kind)}</code>.` : ""}
        </div>
        <div class="hosted-evidence-grid">
          ${(experiment.published_results || []).map((item) => `
            <article class="hosted-evidence-card hosted-published-card">
              <strong title="${escapeAttr(item.title)}">${escapeHtml(item.title)}</strong>
              <span class="evidence-type">published</span>
              <code title="${escapeAttr(item.result_id)}">result ${escapeHtml(item.result_id)}</code>
              <code title="${escapeAttr(item.source_bundle_sha256)}">source bundle sha256 ${escapeHtml(item.source_bundle_sha256)}</code>
              <a href="${escapeAttr(item.path)}">Open generated result page</a>
            </article>
          `).join("")}
          ${experiment.evidence.map((item) => `
            <article class="hosted-evidence-card">
              <strong title="${escapeAttr(item.path)}">${escapeHtml(item.label)}</strong>
              <span class="evidence-type">${escapeHtml(item.type)}</span>
              <code title="${escapeAttr(item.sha256)}">sha256 ${escapeHtml(item.sha256)}</code>
              <code>${escapeHtml(formatBytes(item.size_bytes))} · ${escapeHtml(item.path)}</code>
              <a href="${escapeAttr(item.path)}">Open stored evidence</a>
            </article>
          `).join("")}
        </div>
      </section>
    `;
  }

  function formatBytes(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "unknown size";
    if (number >= 1024 * 1024) return `${(number / (1024 * 1024)).toFixed(2)} MiB`;
    if (number >= 1024) return `${(number / 1024).toFixed(1)} KiB`;
    return `${number} B`;
  }

  function finiteOrDash(value) {
    return Number.isFinite(Number(value)) ? String(value) : "—";
  }

  function renderFatalError(message) {
    elements.resultsComparison.innerHTML = `<div class="hosted-error">${escapeHtml(message)}</div>`;
    elements.recipeGraph.innerHTML = `<div class="hosted-error">${escapeHtml(message)}</div>`;
    showToast(message);
  }

  function showToast(message) {
    elements.toast.textContent = message;
    elements.toast.classList.add("visible");
    window.clearTimeout(showToast.timer);
    showToast.timer = window.setTimeout(() => {
      elements.toast.classList.remove("visible");
    }, 4800);
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function escapeAttr(value) {
    return escapeHtml(value);
  }

  function cssEscape(value) {
    if (window.CSS?.escape) return window.CSS.escape(String(value));
    return String(value).replace(/[^a-zA-Z0-9_-]/g, "\\$&");
  }
})();
