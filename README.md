# data-science-with-python
data science, 빅데이터분석기사 실기, ADP 실기

## Python Version == 3.7.4

### Packages Version
```
numpy 1.18.5
pandas 0.25.1
scikit-learn 0.21.3
scipy 1.5.2
statsmodels 0.11.1
xgboost 0.80

zipp 3.8.1
ydata-profiling 4.12.1
wordcloud 1.9.4
wincertstore 0.2
widgetsnbextension 4.0.3
webencodings 0.5.1
wcwidth 0.2.5
visions 0.7.5
urllib3 2.0.7
typing-extensions 4.3.0
typeguard 4.1.2
traitlets 5.4.0
tqdm 4.67.1
tornado 6.2
tinycss2 1.1.1
terminado 0.15.0
tangled-up-in-unicode 0.2.0
SQLAlchemy 2.0.37
soupsieve 2.3.2.post1
smart-open 6.2.0
six 1.16.0
setuptools 40.8.0
Send2Trash 1.8.0
seaborn 0.12.2
requests 2.31.0
QtPy 2.2.0
qtconsole 5.3.2
pyzmq 24.0.1
PyYAML 6.0.1
pywinpty 2.0.8
pywin32 304
PyWavelets 1.3.0
pytz 2022.2.1
python-dateutil 2.8.2
pyrsistent 0.18.1
pyparsing 3.0.9
Pygments 2.13.0
pydantic 2.5.3
pydantic-core 2.14.6
pycparser 2.21
psutil 5.9.2
prompt-toolkit 3.0.31
prometheus-client 0.14.1
pkgutil-resolve-name 1.3.10
pip 23.0.1
Pillow 9.2.0
pickleshare 0.7.5
phik 0.12.3
patsy 0.5.2
parso 0.8.3
pandocfilters 1.5.0
pandas-profiling 3.6.6
packaging 21.3
optuna 4.0.0
numba 0.56.4
notebook 6.4.12
networkx 2.6.3
nest-asyncio 1.5.5
nbformat 5.6.1
nbconvert 7.0.0
nbclient 0.6.8
multimethod 1.9.1
mlxtend 0.15.0.0
mistune 2.0.4
matplotlib 3.5.3
matplotlib-inline 0.1.6
MarkupSafe 2.1.1
Mako 1.2.4
lxml 4.9.1
llvmlite 0.39.1
lightgbm 4.5.0
kiwisolver 1.4.4
jupyterlab-widgets 3.0.3
jupyterlab-pygments 0.2.2
jupyter 1.0.0
jupyter-core 4.11.1
jupyter-console 6.4.4
jupyter-client 7.3.5
jsonschema 4.16.0
joblib 1.2.0
Jinja2 3.1.2
jedi 0.18.1
ipywidgets 8.0.2
ipython 7.34.0
ipython-genutils 0.2.0
ipykernel 6.16.0
importlib-resources 5.9.0
importlib-metadata 4.12.0
imbalanced-learn 0.5.0
ImageHash 4.3.1
idna 3.10
htmlmin 0.1.12
greenlet 3.1.1
gensim 3.7.1
fonttools 4.37.3
fastjsonschema 2.16.2
entrypoints 0.4
defusedxml 0.7.1
decorator 5.1.1
debugpy 1.6.3
dacite 1.8.1
Cython 0.29.32
cycler 0.11.0
colorlog 6.9.0
colorama 0.4.5
charset-normalizer 3.4.1
cffi 1.15.1
certifi 2024.12.14
bleach 5.0.1
beautifulsoup4 4.11.1
backcall 0.2.0
attrs 22.1.0
argon2-cffi 21.3.0
argon2-cffi-bindings 21.2.0
annotated-types 0.5.0
alembic 1.12.1
```

<br>

### Library
```
import pandas as pd
import numpy as np

from datetime import datetime


from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_distances

# 샘플링
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE 
from imblearn.under_sampling import RandomUnderSampler

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# 앙상블
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

# 이상치 제거
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# 평가 지표
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, auc
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso, LassoLars
import statsmodels.api as sm


from sklearn.cluster import DBSCAN # 이상치 제거
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering


from xgboost import XGBClassifier


from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV
```