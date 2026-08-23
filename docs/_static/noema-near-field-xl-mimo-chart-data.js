(function () {
  "use strict";
  const charts = {
  "near-field-focusing-gain": {
    "accessibleSummary": "Each point is the mean of three paired held-out runs. Shaded bands show Student-t 95% confidence intervals; exact values and run identifiers are downloadable below.",
    "allowLog": false,
    "description": "Mean normalized focusing gain over three paired held-out target-state and noise seeds. True-position focusing is a simulation-only oracle.",
    "series": [
      {
        "color": "#2563eb",
        "dash": [
          7,
          4
        ],
        "id": "far_field_steering",
        "label": "Far-field steering search",
        "marker": "square",
        "range": [
          [
            0.0,
            0.8579891434732572,
            0.9176227339680763
          ],
          [
            15.0,
            0.8851204579500078,
            0.9204530849633255
          ]
        ],
        "values": [
          [
            0.0,
            0.8878059387206667
          ],
          [
            15.0,
            0.9027867714566666
          ]
        ]
      },
      {
        "color": "#7c3aed",
        "dash": [
          10,
          3
        ],
        "id": "polar_codebook",
        "label": "Polar range-angle codebook",
        "marker": "triangle",
        "range": [
          [
            0.0,
            0.8633317800250632,
            0.9050529150282701
          ],
          [
            15.0,
            0.8867579775831939,
            0.9224773965494727
          ]
        ],
        "values": [
          [
            0.0,
            0.8841923475266666
          ],
          [
            15.0,
            0.9046176870663333
          ]
        ]
      },
      {
        "color": "#d97706",
        "dash": [
          3,
          3
        ],
        "id": "oracle_focus",
        "label": "True-position focusing",
        "marker": "diamond",
        "range": [
          [
            0.0,
            1.0,
            1.0
          ],
          [
            15.0,
            1.0,
            1.0
          ]
        ],
        "values": [
          [
            0.0,
            1.0
          ],
          [
            15.0,
            1.0
          ]
        ]
      },
      {
        "color": "#16a34a",
        "dash": [],
        "id": "learned_near_field_estimator",
        "label": "Physics-informed learned estimator",
        "marker": "circle",
        "range": [
          [
            0.0,
            0.9486190067550988,
            0.9675221853629012
          ],
          [
            15.0,
            0.9730258988668122,
            0.9851179711685211
          ]
        ],
        "values": [
          [
            0.0,
            0.958070596059
          ],
          [
            15.0,
            0.9790719350176667
          ]
        ]
      }
    ],
    "title": "Near-field focusing gain",
    "type": "line",
    "xLabel": "SNR (dB)",
    "yDomain": [
      0.0,
      1.05
    ],
    "yIncludeZero": true,
    "yLabel": "Normalized focusing gain"
  }
};
  window.NOEMA_DEMO_CHARTS = Object.freeze({
    ...(window.NOEMA_DEMO_CHARTS || {}),
    ...charts,
  });
})();
