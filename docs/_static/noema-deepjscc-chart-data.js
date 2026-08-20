(function () {
  "use strict";
  const charts = {
  "deepjscc-ms-ssim": {
    "accessibleSummary": "MS-SSIM follows the PSNR ordering: DeepJSCC leads at minus 6 and minus 4 dB, then capacity-matched JPEG leads.",
    "allowLog": false,
    "description": "Capacity-matched JPEG is deterministic. DeepJSCC uses one frozen model; whiskers show a two-sided 95% Student's t interval across three channel-noise seeds for the same four crops.",
    "series": [
      {
        "color": "#2563eb",
        "dash": [
          7,
          4
        ],
        "id": "jpeg_capacity",
        "label": "Capacity-matched JPEG",
        "marker": "square",
        "range": [
          [
            -6.0,
            0.269506890327,
            0.269506890327
          ],
          [
            -4.0,
            0.508743267506,
            0.508743267506
          ],
          [
            -2.0,
            0.870683416724,
            0.870683416724
          ],
          [
            0.0,
            0.928130552173,
            0.928130552173
          ],
          [
            4.0,
            0.972190111876,
            0.972190111876
          ],
          [
            8.0,
            0.985090196133,
            0.985090196133
          ],
          [
            12.0,
            0.990412741899,
            0.990412741899
          ],
          [
            16.0,
            0.99303945899,
            0.99303945899
          ]
        ],
        "sampleCount": 1,
        "values": [
          [
            -6.0,
            0.269506890327
          ],
          [
            -4.0,
            0.508743267506
          ],
          [
            -2.0,
            0.870683416724
          ],
          [
            0.0,
            0.928130552173
          ],
          [
            4.0,
            0.972190111876
          ],
          [
            8.0,
            0.985090196133
          ],
          [
            12.0,
            0.990412741899
          ],
          [
            16.0,
            0.99303945899
          ]
        ]
      },
      {
        "color": "#16a34a",
        "dash": [],
        "id": "learned_deepjscc",
        "label": "Learned DeepJSCC",
        "marker": "circle",
        "range": [
          [
            -6.0,
            0.7405100200930532,
            0.7495578095956135
          ],
          [
            -4.0,
            0.8015160355337843,
            0.8084274735682158
          ],
          [
            -2.0,
            0.8461374224298857,
            0.8509624023161144
          ],
          [
            0.0,
            0.8777987128331798,
            0.881349616663487
          ],
          [
            4.0,
            0.914554371489555,
            0.9164617915417783
          ],
          [
            8.0,
            0.93067856685651,
            0.9316915864121568
          ],
          [
            12.0,
            0.9374111979063829,
            0.9379891664929504
          ],
          [
            16.0,
            0.9401745411895472,
            0.9404912478584526
          ]
        ],
        "sampleCount": 3,
        "values": [
          [
            -6.0,
            0.7450339148443333
          ],
          [
            -4.0,
            0.804971754551
          ],
          [
            -2.0,
            0.848549912373
          ],
          [
            0.0,
            0.8795741647483334
          ],
          [
            4.0,
            0.9155080815156666
          ],
          [
            8.0,
            0.9311850766343334
          ],
          [
            12.0,
            0.9377001821996667
          ],
          [
            16.0,
            0.9403328945239999
          ]
        ]
      }
    ],
    "title": "Perceptual reconstruction quality under AWGN",
    "type": "line",
    "xLabel": "SNR (dB)",
    "yIncludeZero": false,
    "yLabel": "MS-SSIM"
  },
  "deepjscc-psnr": {
    "accessibleSummary": "DeepJSCC has higher mean PSNR at minus 6 and minus 4 dB. Capacity-matched JPEG is higher from minus 2 through 16 dB.",
    "allowLog": false,
    "description": "Capacity-matched JPEG is deterministic. DeepJSCC uses one frozen model; whiskers show a two-sided 95% Student's t interval across three channel-noise seeds for the same four crops.",
    "series": [
      {
        "color": "#2563eb",
        "dash": [
          7,
          4
        ],
        "id": "jpeg_capacity",
        "label": "Capacity-matched JPEG",
        "marker": "square",
        "range": [
          [
            -6.0,
            13.7215445512,
            13.7215445512
          ],
          [
            -4.0,
            17.6373951835,
            17.6373951835
          ],
          [
            -2.0,
            25.3384820424,
            25.3384820424
          ],
          [
            0.0,
            27.9425205181,
            27.9425205181
          ],
          [
            4.0,
            31.3470709649,
            31.3470709649
          ],
          [
            8.0,
            33.8647301587,
            33.8647301587
          ],
          [
            12.0,
            35.8725795691,
            35.8725795691
          ],
          [
            16.0,
            37.5039653338,
            37.5039653338
          ]
        ],
        "sampleCount": 1,
        "values": [
          [
            -6.0,
            13.7215445512
          ],
          [
            -4.0,
            17.6373951835
          ],
          [
            -2.0,
            25.3384820424
          ],
          [
            0.0,
            27.9425205181
          ],
          [
            4.0,
            31.3470709649
          ],
          [
            8.0,
            33.8647301587
          ],
          [
            12.0,
            35.8725795691
          ],
          [
            16.0,
            37.5039653338
          ]
        ]
      },
      {
        "color": "#16a34a",
        "dash": [],
        "id": "learned_deepjscc",
        "label": "Learned DeepJSCC",
        "marker": "circle",
        "range": [
          [
            -6.0,
            21.88273819857969,
            22.10711476262031
          ],
          [
            -4.0,
            23.038192937800968,
            23.237612508265695
          ],
          [
            -2.0,
            23.909934486011156,
            24.10458681792218
          ],
          [
            0.0,
            24.560041517039114,
            24.749823553760887
          ],
          [
            4.0,
            25.363931995980764,
            25.52320736495257
          ],
          [
            8.0,
            25.748417610133497,
            25.861718357999838
          ],
          [
            12.0,
            25.921879499583127,
            25.997268947683537
          ],
          [
            16.0,
            25.998414914608933,
            26.0467138993244
          ]
        ],
        "sampleCount": 3,
        "values": [
          [
            -6.0,
            21.9949264806
          ],
          [
            -4.0,
            23.13790272303333
          ],
          [
            -2.0,
            24.007260651966668
          ],
          [
            0.0,
            24.6549325354
          ],
          [
            4.0,
            25.443569680466666
          ],
          [
            8.0,
            25.805067984066667
          ],
          [
            12.0,
            25.959574223633332
          ],
          [
            16.0,
            26.022564406966666
          ]
        ]
      }
    ],
    "title": "Reconstruction quality at a fixed bandwidth ratio",
    "type": "line",
    "xLabel": "SNR (dB)",
    "yIncludeZero": false,
    "yLabel": "PSNR (dB)"
  }
};
  window.NOEMA_DEMO_CHARTS = Object.freeze({
    ...(window.NOEMA_DEMO_CHARTS || {}),
    ...charts,
  });
})();
