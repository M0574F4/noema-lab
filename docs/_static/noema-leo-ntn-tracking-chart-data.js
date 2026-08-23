(function () {
  "use strict";
  const charts = {
  "leo-ntn-handover-accuracy": {
    "accessibleSummary": "Each point is the mean of three paired held-out runs. Shaded bands show Student-t 95% confidence intervals; exact values and run identifiers are downloadable below.",
    "allowLog": false,
    "description": "Mean one-second-ahead beam accuracy over three paired held-out track and measurement-noise seeds. True future state is a simulation-only oracle.",
    "series": [
      {
        "color": "#2563eb",
        "dash": [
          7,
          4
        ],
        "id": "hold_last",
        "label": "Hold last observation",
        "marker": "square",
        "range": [
          [
            0.0,
            0.47854911922198473,
            0.7922842141113485
          ],
          [
            15.0,
            0.8890486836983789,
            0.9338679829682879
          ]
        ],
        "values": [
          [
            0.0,
            0.6354166666666666
          ],
          [
            15.0,
            0.9114583333333334
          ]
        ]
      },
      {
        "color": "#7c3aed",
        "dash": [
          10,
          3
        ],
        "id": "linear_extrapolation",
        "label": "Linear Doppler/angle extrapolation",
        "marker": "triangle",
        "range": [
          [
            0.0,
            0.10043108425038123,
            0.3058189157496188
          ],
          [
            15.0,
            0.6275350851585606,
            0.8516315815081061
          ]
        ],
        "values": [
          [
            0.0,
            0.203125
          ],
          [
            15.0,
            0.7395833333333334
          ]
        ]
      },
      {
        "color": "#d97706",
        "dash": [
          3,
          3
        ],
        "id": "oracle_future",
        "label": "True future state",
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
        "id": "learned_ntn_tracker",
        "label": "Learned causal tracker",
        "marker": "circle",
        "range": [
          [
            0.0,
            0.6515210510951364,
            0.7859789489048636
          ],
          [
            15.0,
            0.8510140340634242,
            0.9406526326032425
          ]
        ],
        "values": [
          [
            0.0,
            0.71875
          ],
          [
            15.0,
            0.8958333333333334
          ]
        ]
      }
    ],
    "title": "LEO-NTN next-beam handover",
    "type": "line",
    "xLabel": "SNR (dB)",
    "yDomain": [
      0.0,
      1.05
    ],
    "yIncludeZero": true,
    "yLabel": "Next-beam accuracy"
  }
};
  window.NOEMA_DEMO_CHARTS = Object.freeze({
    ...(window.NOEMA_DEMO_CHARTS || {}),
    ...charts,
  });
})();
