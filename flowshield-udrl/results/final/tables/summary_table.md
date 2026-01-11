# Evaluation Summary

Date: 2026-01-11 15:39:31

| Configuration    |   ('Mean Return', 'ID') |   ('Mean Return', 'OOD_extreme') |   ('Mean Return', 'OOD_moderate') |   ('Success Rate (%)', 'ID') |   ('Success Rate (%)', 'OOD_extreme') |   ('Success Rate (%)', 'OOD_moderate') |
|:-----------------|------------------------:|---------------------------------:|----------------------------------:|-----------------------------:|--------------------------------------:|---------------------------------------:|
| Diffusion Shield |                 225.077 |                          245.145 |                           195.34  |                           90 |                                   100 |                                     80 |
| Flow Shield      |                 217.053 |                          222.117 |                           229.452 |                           90 |                                   100 |                                     90 |
| No Shield        |                 231.188 |                          235.072 |                           199.137 |                          100 |                                   100 |                                     80 |
| Quantile Shield  |                 107.038 |                          215.943 |                           189.19  |                           50 |                                    90 |                                     80 |

## Full Results

| Command      | Configuration    |   Mean Return |   Std Return |   Min Return |   Max Return |   Success Rate (%) |   Crash Rate (%) |   Shield Activations |
|:-------------|:-----------------|--------------:|-------------:|-------------:|-------------:|-------------------:|-----------------:|---------------------:|
| ID           | No Shield        |       231.188 |      34.055  |    144.513   |      263.324 |                100 |                0 |                  0   |
| ID           | Flow Shield      |       217.053 |      71.6921 |     11.6659  |      278.897 |                 90 |               10 |                  9.9 |
| ID           | Quantile Shield  |       107.038 |     114.907  |    -21.0627  |      266.78  |                 50 |               50 |                  9.9 |
| ID           | Diffusion Shield |       225.077 |      75.9869 |      2.98399 |      284.729 |                 90 |               10 |                  2.2 |
| OOD_moderate | No Shield        |       199.137 |     108.9    |    -30.9381  |      276.868 |                 80 |               20 |                  0   |
| OOD_moderate | Flow Shield      |       229.452 |      72.6602 |     17.2963  |      275.106 |                 90 |               10 |                 13.1 |
| OOD_moderate | Quantile Shield  |       189.19  |      95.5828 |      4.60839 |      279.216 |                 80 |               20 |                  8.3 |
| OOD_moderate | Diffusion Shield |       195.34  |     108.909  |    -26.5561  |      274.157 |                 80 |               20 |                  1.8 |
| OOD_extreme  | No Shield        |       235.072 |      14.1001 |    206.692   |      252.997 |                100 |                0 |                  0   |
| OOD_extreme  | Flow Shield      |       222.117 |      38.1451 |    137.314   |      269.917 |                100 |                0 |                 15.7 |
| OOD_extreme  | Quantile Shield  |       215.943 |      62.8225 |     43.9392  |      278.86  |                 90 |               10 |                 10.5 |
| OOD_extreme  | Diffusion Shield |       245.145 |      25.3322 |    214.753   |      285.768 |                100 |                0 |                  2.2 |