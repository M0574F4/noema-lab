(function () {
  "use strict";
  const charts = {
  "range-localization-rmse": {
    "accessibleSummary": "Each point is the mean of three paired held-out runs. Shaded bands show Student-t 95% confidence intervals; exact values and run identifiers are downloadable below.",
    "allowLog": false,
    "description": "Mean RMSE over three paired held-out target, range-noise, and geometry seeds. Bands are two-sided Student-t 95% intervals.",
    "series": [
      {
        "color": "#2563eb",
        "dash": [
          7,
          4
        ],
        "id": "trilateration",
        "label": "Linear trilateration",
        "marker": "square",
        "range": [
          [
            0.0,
            4.2802096223355015,
            6.308734812784499
          ],
          [
            15.0,
            0.7776099340688997,
            1.113271246691767
          ]
        ],
        "values": [
          [
            0.0,
            5.29447221756
          ],
          [
            15.0,
            0.9454405903803333
          ]
        ]
      },
      {
        "color": "#7c3aed",
        "dash": [
          10,
          3
        ],
        "id": "regularized_trilateration",
        "label": "Regularized trilateration",
        "marker": "triangle",
        "range": [
          [
            0.0,
            4.043868633044185,
            6.119006542435815
          ],
          [
            15.0,
            0.726352444793592,
            1.2296224122217412
          ]
        ],
        "values": [
          [
            0.0,
            5.08143758774
          ],
          [
            15.0,
            0.9779874285076666
          ]
        ]
      },
      {
        "color": "#16a34a",
        "dash": [],
        "id": "learned_localizer",
        "label": "Learned residual localizer",
        "marker": "circle",
        "range": [
          [
            0.0,
            3.197751244306079,
            5.871494412027253
          ],
          [
            15.0,
            0.8564319589623104,
            1.8932098568576896
          ]
        ],
        "values": [
          [
            0.0,
            4.5346228281666665
          ],
          [
            15.0,
            1.37482090791
          ]
        ]
      }
    ],
    "title": "Range-localization error",
    "type": "line",
    "xLabel": "SNR (dB)",
    "yIncludeZero": true,
    "yLabel": "Position RMSE (m)"
  }
};
  window.NOEMA_DEMO_CHARTS = Object.freeze({
    ...(window.NOEMA_DEMO_CHARTS || {}),
    ...charts,
  });
})();
