# SINDy Markov Model Log: notebook_adaptive.log

*Generated from logs/notebook_adaptive.log*

Initializing SINDy Markov Model


True term indices: [0]


Computing Gram matrix with 200 sample points


Log determinant of Gram matrix: -5.0778


## 🔍 BEGINNING SUCCESS PROBABILITY CALCULATION

All indices = {0, 1, 2}


True indices = {0}


Target state (true model) = {0}


Generated 4 valid states


TRUE MODEL STATE TRANSITIONS


Transitions from {0}:


    -> [STOP]: 1.000000


TRUE MODEL STATE STOPPING PROBABILITY: 1.000000





## ❌ ERROR

Error calculating transition probability: unhashable type: 'numpy.ndarray'


Transitions from {0, 1}:


    -> [STOP]: 1.000000


## ❌ ERROR

Error calculating transition probability: unhashable type: 'numpy.ndarray'


Transitions from {0, 2}:


    -> [STOP]: 1.000000


## ❌ ERROR

Error calculating transition probability: unhashable type: 'numpy.ndarray'


## ❌ ERROR

Error calculating transition probability: unhashable type: 'numpy.ndarray'


## ❌ ERROR

Error calculating transition probability: unhashable type: 'numpy.ndarray'


Transitions from {0, 1, 2}:


    -> [STOP]: 1.000000


## 🔍 SUCCESS PROBABILITY CALCULATION

Method 1 - Direct Calculation:


  Probability of reaching true state:    0.000000


  Probability of stopping at true state: 1.000000


  Success probability = 0.000000 × 1.000000 = 0.000000


## 🔍 END OF SUCCESS PROBABILITY CALCULATION

Running adaptive STLSQ simulation (max 500 trials, 95% confidence, 3.0% margin)


After 30 trials: Success rate = 0.0000, Margin of error = ±0.2198 (target: 0.0300)


After 50 trials: Success rate = 0.0200, Margin of error = ±0.1820 (target: 0.0300)


After 100 trials: Success rate = 0.0100, Margin of error = ±0.1321 (target: 0.0300)


After 150 trials: Success rate = 0.0067, Margin of error = ±0.1089 (target: 0.0300)


After 200 trials: Success rate = 0.0050, Margin of error = ±0.0947 (target: 0.0300)


After 250 trials: Success rate = 0.0080, Margin of error = ±0.0853 (target: 0.0300)


After 300 trials: Success rate = 0.0067, Margin of error = ±0.0780 (target: 0.0300)


After 350 trials: Success rate = 0.0057, Margin of error = ±0.0722 (target: 0.0300)


After 400 trials: Success rate = 0.0050, Margin of error = ±0.0676 (target: 0.0300)


After 450 trials: Success rate = 0.0044, Margin of error = ±0.0638 (target: 0.0300)


After 500 trials: Success rate = 0.0040, Margin of error = ±0.0605 (target: 0.0300)


STLSQ simulation results: 2/500 successful, 0.0040 success rate, margin of error: ±0.0605


