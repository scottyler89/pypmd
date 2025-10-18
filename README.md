# PMD #

### What is this repository for? ###

* Percent Maximum Difference (PMD), is a new relative distance metric for count-data that can linearly quantify how similar/different your observations are based on the composition of their features. It has an upper bound of 1 when completly different, and a centered (not bounded) around 0 . You can also subtract it from 1, to flip this (reverse PMD: rPMD). This can be more intuitive in some ways because it can be thought of as like a correlation: How 'correlated' are my samples with each other based on the composition of their features.

### How do I get set up? ###

This repository is pip installable:
`python3 -m pip install percent_max_diff`

You can also clone the repository and install using the setup.py script in the distribution like so:
`python3 setup.py install`
or
`python3 -m pip intall .`


### How do I use it? ###

```
import numpy as np
from percent_max_diff import pmd
pmd_res = pmd(np.array([[100,0,0],[0,100,0],[0,0,100]]))
print(pmd_res.pmd)
```

### License ###
For non-commercial use, PyMINEr is available via the AGPLv3 license. Commercial entities should inquire with scottyler89@gmail.com

### Who do I talk to? ###

* Repo owner/admin: scottyler89+bitbucket@gmail.com
