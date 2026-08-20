(function () {
  "use strict";

  const ROOT_ID = "noema-break-comparison";
  const SERIES_IDS = {
    baseline: "uncompensated_qpsk",
    reference: "calibrated_iq_oracle",
    candidate: "learned_receiver",
  };
  const FAULTS = ["condition", "aggregation", "metric", "role"];

  function byId(id) {
    return document.getElementById(id);
  }

  function requireValue(condition, message) {
    if (!condition) {
      throw new Error(message);
    }
  }

  function mean(values) {
    return values.reduce(function (total, value) {
      return total + value;
    }, 0) / values.length;
  }

  function formatRate(value) {
    if (!Number.isFinite(value)) {
      return "—";
    }
    if (value === 0) {
      return "0";
    }
    if (value < 0.01) {
      return value.toExponential(3).replace("e-", "e−");
    }
    return value.toFixed(4);
  }

  function formatPercent(value) {
    return Number.isFinite(value) ? Math.abs(value).toFixed(2) + "%" : "—";
  }

  function formatSnr(value) {
    return Number(value).toLocaleString(undefined, { maximumFractionDigits: 1 }) + " dB";
  }

  function pointFor(series, snr) {
    const point = series.points.find(function (candidate) {
      return Number(candidate.snr_db) === Number(snr);
    });
    requireValue(point, "Missing evidence point at " + snr + " dB for " + series.id);
    requireValue(point.summary && Array.isArray(point.observations), "Incomplete point for " + series.id);
    return point;
  }

  function validateEvidence(data) {
    requireValue(data && data.kind === "noema.launch_evidence_projection", "Unexpected evidence kind");
    requireValue(data.design && Array.isArray(data.design.snr_db), "Missing experiment design");
    requireValue(data.design.snr_db.length > 1, "The demo needs at least two SNR cells");
    requireValue(data.headline && Number.isFinite(data.headline.primary_snr_db), "Missing headline cell");
    requireValue(Array.isArray(data.series), "Missing evidence series");
    Object.keys(SERIES_IDS).forEach(function (role) {
      const series = data.series.find(function (candidate) {
        return candidate.id === SERIES_IDS[role];
      });
      requireValue(series && Array.isArray(series.points), "Missing " + SERIES_IDS[role] + " series");
      data.design.snr_db.forEach(function (snr) {
        const point = pointFor(series, snr);
        requireValue(point.observations.length > 0, "Missing observations for " + series.id);
      });
    });
    requireValue(data.scientific_status && data.scientific_status.disclosure, "Missing status disclosure");
    requireValue(typeof data.sha256 === "string" && data.sha256.length === 64, "Missing evidence identity");
  }

  function initialize(root, data) {
    validateEvidence(data);

    const series = {};
    Object.keys(SERIES_IDS).forEach(function (role) {
      series[role] = data.series.find(function (candidate) {
        return candidate.id === SERIES_IDS[role];
      });
    });

    const snrSelect = byId("btc-snr");
    const fieldset = root.querySelector(".noema-btc__faults");
    const reset = byId("btc-reset");
    snrSelect.replaceChildren();
    data.design.snr_db.forEach(function (snr) {
      const option = document.createElement("option");
      option.value = String(snr);
      option.textContent = formatSnr(snr);
      snrSelect.appendChild(option);
    });
    const defaultIndex = data.design.snr_db.findIndex(function (snr) {
      return Number(snr) === Number(data.headline.primary_snr_db);
    });
    requireValue(defaultIndex >= 0, "The headline SNR is outside the declared design");
    const defaultSnr = Number(data.design.snr_db[defaultIndex]);
    snrSelect.value = String(defaultSnr);

    byId("btc-run-count").textContent =
      data.design.run_count + " retained runs · " + data.design.snr_db.length + " SNR cells";
    byId("btc-evidence-hash").textContent = "Evidence " + data.sha256.slice(0, 12) + "…";
    byId("btc-disclosure").textContent = data.scientific_status.disclosure;
    const evidenceLink = byId("btc-evidence-link");
    evidenceLink.href = root.dataset.evidenceUrl;

    function activeFaults() {
      return FAULTS.filter(function (fault) {
        return byId("btc-fault-" + fault).checked;
      });
    }

    function adjacentHigherSnr(selected) {
      const current = data.design.snr_db.findIndex(function (value) {
        return Number(value) === Number(selected);
      });
      return data.design.snr_db[current + 1] !== undefined
        ? data.design.snr_db[current + 1]
        : data.design.snr_db[current - 1];
    }

    function setCheck(id, valid) {
      const row = byId("btc-check-" + id);
      row.dataset.check = valid ? "pass" : "fail";
      const icon = row.firstElementChild;
      icon.textContent = valid ? "✓" : "×";
      icon.setAttribute("aria-label", valid ? "Pass" : "Fail");
    }

    function render() {
      const faults = activeFaults();
      const broken = faults.length > 0;
      const selectedSnr = Number(snrSelect.value);
      const candidateSnr = faults.includes("condition")
        ? adjacentHigherSnr(selectedSnr)
        : selectedSnr;
      const baselineSeries = faults.includes("role") ? series.reference : series.baseline;
      const baselinePoint = pointFor(baselineSeries, selectedSnr);
      const candidatePoint = pointFor(series.candidate, candidateSnr);
      const baselineValue = baselinePoint.summary.mean_ber;
      const candidateMetric = faults.includes("metric") ? "BLER" : "BER";
      const candidateValues = candidatePoint.observations.map(function (observation) {
        return faults.includes("metric") ? observation.bler : observation.ber;
      });
      const candidateValue = faults.includes("aggregation")
        ? Math.min.apply(null, candidateValues)
        : mean(candidateValues);
      const comparable = !faults.includes("metric");
      const change = comparable ? ((baselineValue - candidateValue) / baselineValue) * 100 : NaN;

      root.dataset.state = broken ? "broken" : "intact";
      const verdict = byId("btc-verdict");
      verdict.dataset.verdict = broken ? "broken" : "intact";
      byId("btc-result-title").textContent = broken ? "Comparison broken" : "Contract intact";
      byId("btc-verdict-detail").textContent = broken
        ? "The result still looks precise, but " + faults.length + " declared rule" + (faults.length === 1 ? " is" : "s are") + " no longer satisfied."
        : "Both methods share the declared condition, seeds, aggregation, metric, and comparator roles.";

      byId("btc-baseline-label").textContent = baselineSeries.label;
      byId("btc-baseline-value").textContent = formatRate(baselineValue);
      byId("btc-baseline-meta").textContent =
        "Mean BER · " + formatSnr(selectedSnr) + " · " + baselinePoint.observations.length + " seeds";
      byId("btc-candidate-label").textContent = series.candidate.label;
      byId("btc-candidate-value").textContent = formatRate(candidateValue);
      byId("btc-candidate-meta").textContent =
        (faults.includes("aggregation") ? "Best-seed " : "Mean ") +
        candidateMetric + " · " + formatSnr(candidateSnr) + " · " +
        (faults.includes("aggregation") ? "1 selected seed" : candidatePoint.observations.length + " seeds");

      const claimValue = byId("btc-claim-value");
      const claimText = byId("btc-claim-text");
      if (!comparable) {
        claimValue.textContent = "Not comparable";
        claimText.textContent = "BER and BLER have different events and denominators; no relative percentage is defensible.";
      } else {
        claimValue.textContent = formatPercent(change);
        const direction = change >= 0 ? "lower" : "higher";
        claimText.textContent =
          series.candidate.label + " appears " + formatPercent(change) + " " + direction + " than " +
          baselineSeries.label + ".";
      }

      const maxValue = Math.max(baselineValue, candidateValue);
      const baselineWidth = maxValue > 0 ? Math.max(2, (baselineValue / maxValue) * 100) : 2;
      const candidateWidth = maxValue > 0 ? Math.max(2, (candidateValue / maxValue) * 100) : 2;
      byId("btc-baseline-bar").style.width = baselineWidth + "%";
      byId("btc-candidate-bar").style.width = candidateWidth + "%";
      byId("btc-baseline-bar-label").textContent = baselineSeries.label + " · BER";
      byId("btc-candidate-bar-label").textContent = series.candidate.label + " · " + candidateMetric;
      byId("btc-bar-note").textContent = comparable
        ? "Lower is better. Both bars use the same pre-decoder BER scale."
        : "These bars do not share a metric. Similar geometry cannot make BER and BLER comparable.";

      setCheck("condition", !faults.includes("condition"));
      setCheck("aggregation", !faults.includes("aggregation"));
      setCheck("metric", !faults.includes("metric"));
      setCheck("role", !faults.includes("role"));

      const diagnosisTitle = byId("btc-diagnosis-title");
      const diagnosis = byId("btc-diagnosis-list");
      diagnosis.replaceChildren();
      const messages = [];
      if (!broken) {
        diagnosisTitle.textContent = "A narrow claim survives";
        messages.push(
          "The arithmetic supports a within-cell mean-BER comparison over the declared paired seeds.",
          "It remains completed experimental evidence—not a confidence interval, universal guarantee, or publication-ready benchmark."
        );
      } else {
        diagnosisTitle.textContent = "Repair the contract before repeating the claim";
        if (faults.includes("condition")) {
          messages.push(
            "Condition mismatch: the comparator is at " + formatSnr(selectedSnr) +
            " while the candidate is at " + formatSnr(candidateSnr) + ". Return both to the same predeclared cell."
          );
        }
        if (faults.includes("aggregation")) {
          messages.push(
            "Selection bias: the candidate uses its lowest observed error while the comparator uses a three-seed mean. Apply one predeclared aggregation to both sides."
          );
        }
        if (faults.includes("metric")) {
          messages.push(
            "Metric mismatch: a bit error and a block error are different events with different denominators. Compare BER with BER or BLER with BLER."
          );
        }
        if (faults.includes("role")) {
          messages.push(
            "Information mismatch: the calibrated oracle has diagnostic calibration knowledge. Restore its reference label and do not present it as a same-information competitor."
          );
        }
      }
      messages.forEach(function (message) {
        const item = document.createElement("li");
        item.textContent = message;
        diagnosis.appendChild(item);
      });
      reset.disabled = !broken && selectedSnr === defaultSnr;
    }

    root.querySelectorAll("input[data-fault]").forEach(function (input) {
      input.addEventListener("change", render);
    });
    snrSelect.addEventListener("change", render);
    reset.addEventListener("click", function () {
      root.querySelectorAll("input[data-fault]").forEach(function (input) {
        input.checked = false;
      });
      snrSelect.value = String(defaultSnr);
      render();
      snrSelect.focus();
    });

    snrSelect.disabled = false;
    fieldset.disabled = false;
    root.setAttribute("aria-busy", "false");
    render();
  }

  function fail(root, error) {
    root.dataset.state = "error";
    root.setAttribute("aria-busy", "false");
    const verdict = byId("btc-verdict");
    verdict.dataset.verdict = "broken";
    byId("btc-result-title").textContent = "Evidence unavailable";
    byId("btc-verdict-detail").textContent =
      "The demo stopped safely because the canonical evidence could not be validated: " + error.message;
  }

  function boot() {
    const root = byId(ROOT_ID);
    if (!root) {
      return;
    }
    const evidenceUrl = new URL(root.dataset.evidenceUrl, document.baseURI);
    fetch(evidenceUrl, { cache: "no-cache" })
      .then(function (response) {
        if (!response.ok) {
          throw new Error("HTTP " + response.status);
        }
        return response.json();
      })
      .then(function (data) {
        initialize(root, data);
      })
      .catch(function (error) {
        fail(root, error);
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
