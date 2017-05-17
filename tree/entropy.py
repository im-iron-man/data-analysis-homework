def entropy(Y):
    '''calculate the entropy of table'''

    import pandas as pd
    import math

    y = pd.Series(Y)
    y_counts = y.value_counts()
    y_pct = y_counts.div(y_counts.sum().astype(float))
    y_pctarr = y_pct.values

    H = 0
    for i in y_pctarr:
        H += -i * math.log(i, 2)

    return H


Y = [1,1,0,0,0]
print entropy.__name__
print entropy.__doc__
print entropy(Y)
