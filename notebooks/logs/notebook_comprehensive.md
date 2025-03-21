# SINDy Markov Model Log: notebook_comprehensive.log

*Generated from logs/notebook_comprehensive.log*

Initializing SINDy Markov Model


True term indices: [0 1]


Computing Gram matrix with 200 sample points


Log determinant of Gram matrix: 9.9105


## 🔍 BEGINNING SUCCESS PROBABILITY CALCULATION

All indices = {0, 1, 2, 3}


True indices = {0, 1}


Target state (true model) = {0, 1}


Generated 4 valid states


TRUE MODEL STATE TRANSITIONS


Transitions from {0, 1}:


    -> [STOP]: 1.000000


TRUE MODEL STATE STOPPING PROBABILITY: 1.000000





## ❌ ERROR

Error calculating transition probability: unhashable type: 'numpy.ndarray'


Transitions from {0, 1, 2}:


    -> [STOP]: 1.000000


## ❌ ERROR

Error calculating transition probability: unhashable type: 'numpy.ndarray'


Transitions from {0, 1, 3}:


    -> [STOP]: 1.000000


## ❌ ERROR

Error calculating transition probability: unhashable type: 'numpy.ndarray'


## ❌ ERROR

Error calculating transition probability: unhashable type: 'numpy.ndarray'


## ❌ ERROR

Error calculating transition probability: unhashable type: 'numpy.ndarray'


Transitions from {0, 1, 2, 3}:


    -> [STOP]: 1.000000


## 🔍 SUCCESS PROBABILITY CALCULATION

Method 1 - Direct Calculation:


  Probability of reaching true state:    0.000000


  Probability of stopping at true state: 1.000000


  Success probability = 0.000000 × 1.000000 = 0.000000


## 🔍 END OF SUCCESS PROBABILITY CALCULATION

Results - Theoretical: 0.0000, Empirical: 0.1667, Discrepancy: 0.1667


2025-03-20 22:40:28,023 - PIL.PngImagePlugin - DEBUG - STREAM b'IHDR' 16 13


2025-03-20 22:40:28,023 - PIL.PngImagePlugin - DEBUG - STREAM b'tEXt' 41 57


2025-03-20 22:40:28,024 - PIL.PngImagePlugin - DEBUG - STREAM b'pHYs' 110 9


2025-03-20 22:40:28,024 - PIL.PngImagePlugin - DEBUG - STREAM b'IDAT' 131 65536


Results saved to results/sindy_markov_analysis_results.json


