(function () {
  "use strict";
  const charts = {
  "isac-joint-allocation-utility": {
    "accessibleSummary": "Each point is the mean of three paired held-out runs. Shaded bands show Student-t 95% confidence intervals; exact values and run identifiers are downloadable below.",
    "allowLog": false,
    "description": "Mean scalarized utility over three paired held-out communication-channel, sensing-channel, and noise seeds. The number is specific to the contract's fixed sensing weight.",
    "series": [
      {
        "color": "#2563eb",
        "dash": [
          7,
          4
        ],
        "id": "equal_power",
        "label": "Equal power",
        "marker": "square",
        "range": [
          [
            0.0,
            0.3612281450571742,
            0.38069729634149246
          ],
          [
            15.0,
            2.8041637170137648,
            2.914765794972902
          ]
        ],
        "values": [
          [
            0.0,
            0.37096272069933334
          ],
          [
            15.0,
            2.8594647559933333
          ]
        ]
      },
      {
        "color": "#7c3aed",
        "dash": [
          10,
          3
        ],
        "id": "communications_water_filling",
        "label": "Communication-only water filling",
        "marker": "triangle",
        "range": [
          [
            0.0,
            0.38491348296909883,
            0.45545014678290113
          ],
          [
            15.0,
            2.863750323389672,
            2.9432111722503285
          ]
        ],
        "values": [
          [
            0.0,
            0.420181814876
          ],
          [
            15.0,
            2.90348074782
          ]
        ]
      },
      {
        "color": "#d97706",
        "dash": [
          3,
          3
        ],
        "id": "scalarized_reference",
        "label": "Per-scene scalarized optimization",
        "marker": "diamond",
        "range": [
          [
            0.0,
            0.49515127519525537,
            0.5285629642540781
          ],
          [
            15.0,
            2.949603297276541,
            3.0303446782501258
          ]
        ],
        "values": [
          [
            0.0,
            0.5118571197246667
          ],
          [
            15.0,
            2.9899739877633333
          ]
        ]
      },
      {
        "color": "#16a34a",
        "dash": [],
        "id": "learned_isac_allocator",
        "label": "Learned joint allocator",
        "marker": "circle",
        "range": [
          [
            0.0,
            0.49573067134096127,
            0.5281895143530386
          ],
          [
            15.0,
            2.942489531246849,
            3.0237865937864847
          ]
        ],
        "values": [
          [
            0.0,
            0.511960092847
          ],
          [
            15.0,
            2.983138062516667
          ]
        ]
      }
    ],
    "title": "Joint communication-sensing utility",
    "type": "line",
    "xLabel": "SNR (dB)",
    "yIncludeZero": true,
    "yLabel": "Scalarized utility"
  }
};
  window.NOEMA_DEMO_CHARTS = Object.freeze({
    ...(window.NOEMA_DEMO_CHARTS || {}),
    ...charts,
  });
})();
