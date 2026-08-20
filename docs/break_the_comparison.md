# Break the comparison

The percentage is not the experiment. The contract around it is.

This interactive evidence lab starts with a paired, like-for-like receiver comparison. Change one
assumption and watch a plausible-looking claim become scientifically indefensible. Every displayed
measurement is read at runtime from Noema's [canonical launch evidence](launch_evidence.md); the
page does not carry a second copy of the results.

```{raw} html
<div
  id="noema-break-comparison"
  class="noema-btc"
  data-evidence-url="launch_evidence.json"
  data-state="loading"
  aria-busy="true"
>
  <header class="noema-btc__hero">
    <div>
      <p class="noema-btc__eyebrow">Interactive evidence lab</p>
      <h2>Break the comparison.</h2>
      <p class="noema-btc__lede">
        Same retained evidence. One hidden change. A claim that should no longer survive.
      </p>
    </div>
    <div class="noema-btc__provenance" aria-label="Evidence provenance">
      <span id="btc-run-count">Loading run inventory…</span>
      <span id="btc-evidence-hash">Checking evidence…</span>
    </div>
  </header>

  <div class="noema-btc__workspace">
    <aside class="noema-btc__controls" aria-labelledby="btc-controls-title">
      <div class="noema-btc__section-heading">
        <span class="noema-btc__step">01</span>
        <div>
          <p class="noema-btc__kicker">Comparison controls</p>
          <h3 id="btc-controls-title">Change the rules</h3>
        </div>
      </div>

      <label class="noema-btc__select-label" for="btc-snr">
        Declared SNR cell
        <select id="btc-snr" disabled>
          <option>Loading evidence…</option>
        </select>
      </label>

      <fieldset class="noema-btc__faults" disabled>
        <legend>Introduce a hidden fault</legend>

        <label class="noema-btc__fault" for="btc-fault-condition">
          <span>
            <strong>Move the candidate</strong>
            <small>Use the learned result from a different SNR cell.</small>
          </span>
          <input id="btc-fault-condition" type="checkbox" data-fault="condition">
          <span class="noema-btc__switch" aria-hidden="true"></span>
        </label>

        <label class="noema-btc__fault" for="btc-fault-aggregation">
          <span>
            <strong>Cherry-pick a seed</strong>
            <small>Compare the candidate's best seed with the baseline mean.</small>
          </span>
          <input id="btc-fault-aggregation" type="checkbox" data-fault="aggregation">
          <span class="noema-btc__switch" aria-hidden="true"></span>
        </label>

        <label class="noema-btc__fault" for="btc-fault-metric">
          <span>
            <strong>Swap the metric</strong>
            <small>Put candidate BLER beside baseline BER.</small>
          </span>
          <input id="btc-fault-metric" type="checkbox" data-fault="metric">
          <span class="noema-btc__switch" aria-hidden="true"></span>
        </label>

        <label class="noema-btc__fault" for="btc-fault-role">
          <span>
            <strong>Hide privileged information</strong>
            <small>Present the calibrated oracle as an ordinary competitor.</small>
          </span>
          <input id="btc-fault-role" type="checkbox" data-fault="role">
          <span class="noema-btc__switch" aria-hidden="true"></span>
        </label>
      </fieldset>

      <button id="btc-reset" class="noema-btc__reset" type="button" disabled>
        Restore declared comparison
      </button>
    </aside>

    <main class="noema-btc__result" aria-labelledby="btc-result-title">
      <div class="noema-btc__verdict" id="btc-verdict" data-verdict="loading">
        <div class="noema-btc__verdict-icon" aria-hidden="true"><span></span></div>
        <div aria-live="polite" aria-atomic="true">
          <p class="noema-btc__kicker">Contract verdict</p>
          <h3 id="btc-result-title">Loading canonical evidence…</h3>
          <p id="btc-verdict-detail">The controls will unlock when the evidence contract is available.</p>
        </div>
      </div>

      <section class="noema-btc__comparison" aria-label="Current comparison">
        <article class="noema-btc__metric-card noema-btc__metric-card--baseline">
          <p class="noema-btc__metric-role">Comparator</p>
          <h4 id="btc-baseline-label">—</h4>
          <strong id="btc-baseline-value">—</strong>
          <p id="btc-baseline-meta">—</p>
        </article>

        <div class="noema-btc__claim">
          <span>Claim</span>
          <strong id="btc-claim-value">—</strong>
          <p id="btc-claim-text">Waiting for evidence</p>
        </div>

        <article class="noema-btc__metric-card noema-btc__metric-card--candidate">
          <p class="noema-btc__metric-role">Candidate</p>
          <h4 id="btc-candidate-label">—</h4>
          <strong id="btc-candidate-value">—</strong>
          <p id="btc-candidate-meta">—</p>
        </article>
      </section>

      <section class="noema-btc__bars" aria-label="Metric magnitude comparison">
        <div>
          <span id="btc-baseline-bar-label">Comparator</span>
          <span class="noema-btc__bar-track"><span id="btc-baseline-bar"></span></span>
        </div>
        <div>
          <span id="btc-candidate-bar-label">Candidate</span>
          <span class="noema-btc__bar-track"><span id="btc-candidate-bar"></span></span>
        </div>
        <p id="btc-bar-note">Lower is better. Bar lengths share one scale only when the metric contract is intact.</p>
      </section>

      <section class="noema-btc__audit" aria-labelledby="btc-audit-title">
        <div class="noema-btc__section-heading">
          <span class="noema-btc__step">02</span>
          <div>
            <p class="noema-btc__kicker">Live contract audit</p>
            <h3 id="btc-audit-title">What the percentage depends on</h3>
          </div>
        </div>
        <ul class="noema-btc__checks">
          <li id="btc-check-condition"><span aria-hidden="true"></span><div><strong>Same channel condition</strong><small>Both methods use one predeclared SNR cell.</small></div></li>
          <li id="btc-check-aggregation"><span aria-hidden="true"></span><div><strong>Same paired aggregation</strong><small>Both sides use the arithmetic mean over the same seeds.</small></div></li>
          <li id="btc-check-metric"><span aria-hidden="true"></span><div><strong>Same metric and denominator</strong><small>Pre-decoder BER is compared with pre-decoder BER.</small></div></li>
          <li id="btc-check-role"><span aria-hidden="true"></span><div><strong>Honest comparator role</strong><small>Diagnostic references remain distinct from deployable competitors.</small></div></li>
        </ul>
      </section>

      <section class="noema-btc__diagnosis" aria-labelledby="btc-diagnosis-title">
        <p class="noema-btc__kicker">Why it matters</p>
        <h3 id="btc-diagnosis-title">The comparison is loading</h3>
        <ul id="btc-diagnosis-list"><li>Reading the retained observations and declared roles.</li></ul>
      </section>
    </main>
  </div>

  <footer class="noema-btc__boundary">
    <strong>Experimental boundary</strong>
    <p id="btc-disclosure">Loading the scientific-status disclosure…</p>
    <a id="btc-evidence-link" href="launch_evidence.json">Inspect the canonical evidence JSON <span aria-hidden="true">↗</span></a>
  </footer>
</div>

<noscript>
  This demonstration needs JavaScript to recompute the comparison. The complete retained values and
  status disclosure remain available in <a href="launch_evidence.json">launch_evidence.json</a>.
</noscript>
```

The intact state is a teaching example backed by completed experimental evidence, not a declaration
that the benchmark is publication-ready. Observed bands are minima and maxima across three paired
held-out seeds; they are not confidence intervals. See [how the launch evidence is generated and
verified](launch_evidence.md).

## Canonical experiment figure and table

```{figure} _static/launch/f3-receiver-ber-vs-snr.svg
:alt: Pre-decoder BER versus SNR for uncompensated QPSK, a calibrated diagnostic reference, and the learned receiver with an observed min-max band
:class: noema-launch-figure

The static evidence view paired with this interactive audit. Bands are observed minima and maxima,
not confidence intervals.
```

Open the [canonical result table](_static/launch/tables/t0-result-summary.csv) or the complete
[F0–F5 figure and T0–T2 table set](launch_assets.md).

## Watch the complete workflow

The recording continues from this evidence lab through a clean installation, training-contract
export, external training, returned-model validation, and the three-method UI comparison.

```{include} _includes/launch_video.md
```
