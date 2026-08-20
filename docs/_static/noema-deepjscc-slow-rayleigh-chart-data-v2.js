(function () {
  "use strict";
  const charts = {
  "deepjscc-slow-ms-ssim-snr": {
    "allowLog": false,
    "description": "MS-SSIM at \u03ba=0.5 using the same held-out images and paired fades as the PSNR comparison.",
    "series": [
      {
        "color": "#2563eb",
        "dash": [
          7,
          4
        ],
        "id": "digital",
        "label": "JPEG + ideal separation",
        "marker": "square",
        "range": [
          [
            0.0,
            0.4100253940489118,
            0.4594603194924215
          ],
          [
            5.0,
            0.4303506365030037,
            0.46219324314499627
          ],
          [
            10.0,
            0.4352262981501577,
            0.462866585178509
          ],
          [
            15.0,
            0.4365578802372207,
            0.46351072818611266
          ],
          [
            20.0,
            0.4371474661027272,
            0.46386639251327283
          ]
        ],
        "values": [
          [
            0.0,
            0.43474285677066665
          ],
          [
            5.0,
            0.446271939824
          ],
          [
            10.0,
            0.44904644166433333
          ],
          [
            15.0,
            0.4500343042116667
          ],
          [
            20.0,
            0.450506929308
          ]
        ]
      },
      {
        "color": "#16a34a",
        "dash": [],
        "id": "learned",
        "label": "Blind DeepJSCC",
        "marker": "circle",
        "range": [
          [
            0.0,
            0.628777984377918,
            0.742326832218082
          ],
          [
            5.0,
            0.7154950926863365,
            0.8034944901703301
          ],
          [
            10.0,
            0.7748380395129979,
            0.8238860773683354
          ],
          [
            15.0,
            0.8065762177263576,
            0.8264858667889758
          ],
          [
            20.0,
            0.8183017446254657,
            0.8275423572805343
          ]
        ],
        "values": [
          [
            0.0,
            0.685552408298
          ],
          [
            5.0,
            0.7594947914283333
          ],
          [
            10.0,
            0.7993620584406667
          ],
          [
            15.0,
            0.8165310422576667
          ],
          [
            20.0,
            0.822922050953
          ]
        ]
      }
    ],
    "title": "Perceptual quality under slow fading",
    "type": "line",
    "xLabel": "Average SNR (dB)",
    "yIncludeZero": false,
    "yLabel": "MS-SSIM"
  },
  "deepjscc-slow-psnr-snr": {
    "allowLog": false,
    "description": "Both methods use \u03ba=0.5 and paired per-image Rayleigh gains. Bands are two-sided Student-t 95% intervals over three held-out channel seeds.",
    "series": [
      {
        "color": "#2563eb",
        "dash": [
          7,
          4
        ],
        "id": "digital",
        "label": "JPEG + ideal separation",
        "marker": "square",
        "range": [
          [
            0.0,
            15.696915901322399,
            20.88009290654427
          ],
          [
            5.0,
            16.53504596325349,
            22.111424894546513
          ],
          [
            10.0,
            17.329802304350334,
            22.689746918449668
          ],
          [
            15.0,
            18.028458965373673,
            22.989409265826325
          ],
          [
            20.0,
            18.63297062182894,
            23.106684836504392
          ]
        ],
        "values": [
          [
            0.0,
            18.288504403933334
          ],
          [
            5.0,
            19.323235428900002
          ],
          [
            10.0,
            20.0097746114
          ],
          [
            15.0,
            20.5089341156
          ],
          [
            20.0,
            20.869827729166666
          ]
        ]
      },
      {
        "color": "#16a34a",
        "dash": [],
        "id": "learned",
        "label": "Blind DeepJSCC",
        "marker": "circle",
        "range": [
          [
            0.0,
            19.28681550155016,
            20.73745961244984
          ],
          [
            5.0,
            20.45676427218121,
            21.038471629018787
          ],
          [
            10.0,
            20.978298868168636,
            21.107800062364696
          ],
          [
            15.0,
            21.005688852182022,
            21.181938750151314
          ],
          [
            20.0,
            20.983631480858445,
            21.22150406920822
          ]
        ],
        "values": [
          [
            0.0,
            20.012137557
          ],
          [
            5.0,
            20.7476179506
          ],
          [
            10.0,
            21.043049465266666
          ],
          [
            15.0,
            21.093813801166668
          ],
          [
            20.0,
            21.102567775033332
          ]
        ]
      }
    ],
    "title": "Reconstruction quality under slow fading",
    "type": "line",
    "xLabel": "Average SNR (dB)",
    "yIncludeZero": false,
    "yLabel": "PSNR (dB)"
  },
  "deepjscc-slow-rate-psnr": {
    "allowLog": false,
    "description": "The three exported rates come from one nested checkpoint. Each \u03ba cell averages three independently seeded fading replicates; bands expose the resulting outage variance.",
    "series": [
      {
        "color": "#2563eb",
        "dash": [
          7,
          4
        ],
        "id": "digital",
        "label": "JPEG + ideal separation",
        "marker": "square",
        "range": [
          [
            0.125,
            11.520175530783623,
            32.649425836683044
          ],
          [
            0.25,
            0.9269580250767682,
            37.321827085656565
          ],
          [
            0.5,
            17.329802304350334,
            22.689746918449668
          ]
        ],
        "values": [
          [
            0.125,
            22.084800683733334
          ],
          [
            0.25,
            19.12439255536667
          ],
          [
            0.5,
            20.0097746114
          ]
        ]
      },
      {
        "color": "#16a34a",
        "dash": [],
        "id": "learned",
        "label": "Blind DeepJSCC",
        "marker": "circle",
        "range": [
          [
            0.125,
            18.15302375635915,
            22.131113484574183
          ],
          [
            0.25,
            19.22625358714811,
            21.73019423165189
          ],
          [
            0.5,
            20.978298868168636,
            21.107800062364696
          ]
        ],
        "values": [
          [
            0.125,
            20.142068620466667
          ],
          [
            0.25,
            20.4782239094
          ],
          [
            0.5,
            21.043049465266666
          ]
        ]
      }
    ],
    "title": "Rate\u2013distortion slice at 10 dB",
    "type": "line",
    "xLabel": "\u03ba (complex channel uses/source pixel)",
    "yIncludeZero": false,
    "yLabel": "PSNR (dB)"
  }
};
  window.NOEMA_DEMO_CHARTS = Object.freeze({
    ...(window.NOEMA_DEMO_CHARTS || {}),
    ...charts,
  });
})();
