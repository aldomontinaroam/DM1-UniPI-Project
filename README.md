
# Spotify Tracks Dataset
📚 Final project for the first module (Foundations) of the [Data Mining](https://esami.unipi.it/programma.php?c=60963&aa=2023&cid=341&did=13) course, [Università di Pisa](https://didattica.di.unipi.it/en/master-programme-in-data-science-and-business-informatics/).
![images/Firefly 20240128124842.png]
## Dataset
📊 The Spotify Tracks Dataset used in this study was provided by the lecturers and contains information about audio tracks available in the Spotify catalog. These tracks span 20 different genres, such as chicago-house, black-metal and breakbeat. Each track is described by essential details (track’s name, artist, album name, ...) and other features like its level of popularity within the Spotify catalog. The dataset also contains audio-derived features representing various aspects like danceability, energy, key, and loudness.
## Tasks
1. **Data Understanding and Preparation**: exploratory data analysis with the analytical tools studied;  data semantics, assessing data quality, the distribution of the variables and the pairwise correlations.
2. **Clustering**: explore the dataset using centroid-based methods, density-based clustering and hierarchical clustering. 
3. **Classification**: choice of at least target variable and classification by decision trees, KNN and Naive Bayes models. The final discussion must contain the evaluation of the quantitative performance w.r.t. confusion matrix, accuracy, precision, recall, F1-score and ROC-curves.
4. **Regression** and **Pattern Mining**: univariate and multivariate regression techniques choosing 2 or more continuous variables and using different regressor studied; frequent pattern extraction and association rules extration with discussion of the results.

## Language and Packages
- Language: **Python 3.10**
- IDE: [**Google Colab**](https://colab.research.google.com/), cloud-based platform that provides free computational resources, such as CPUs and GPUs, along with tools for writing, running, and sharing Python code.

Below are the Python packages and modules used in the project, categorized by their primary use, with their modules listed:

**General Purpose:**
- `numpy`
- `random`
- `statistics`
- `warnings`

**Data Manipulation:**
- `pandas`
- `scipy`: stats, spatial.distance, spatial
- `sklearn`: preprocessing, feature_selection, model_selection, neighbors, tree, linear_model, naive_bayes, metrics

**Data Visualization:**
- `matplotlib`: pyplot, colors, cm, font_manager
- `seaborn`
- `plotly`: graph_objects, express, subplots
- `mpl_toolkits`: mplot3d
- `scikitplot`: metrics
- `dtreeviz`
- `graphviz`
- `treeinterpreter`

**Machine Learning:**
- `sklearn`: cluster, decomposition, metrics.pairwise, neighbors, tree, model_selection, linear_model, naive_bayes, metrics
- `kneed`

**File and Data Management:**
- `pickle`
- `joblib`
- `google.colab`
- `tqdm`: notebook

**Other:**
- `fim`: apriori, fpgrowth

Final report (PDF) -> [ProjectReport](Project_Argento_Lattanzi_Montinaro.pdf)

## Results
### Clustering
DBSCAN is unable to provide optimal clustering, despite having tested several choices of eps and minPts, because it results mainly in large clusters that include almost the entire dataset, then only noise points; even the hierarchical methods produce highly unbalanced clusters. **K-Means**, applied to a dataset with **selected features**, proved to be the only algorithm capable of separating some clusters in a balanced way with an acceptable silhouette value (0.51).
### Classification
Target variable `genre`:
- **KNN**: `[accuracy: 0.48, roc auc: 0.89, precision/recall auc: 0.44]`. Although we have improved the basic model and we are above the expected value of an accuracy of 1/20, for pure analytical purposes we can consider the model acceptable but not usable in a real-world context, given the high error rate: about half of the data are not classified correctly.
- **Naive Bayes**: In this case the error increases to about 60%, which means that the two models Gaussian and Categorical (on different feature groups, continuous and categor- ical) still perform worse than KNN.
- **DecisionTree**: The accuracy of the model does not exceed 0.46, even after appropriate parameter tuning. However, we can still study the behavior of the model and how it was able to capture relationships between variables based on the importance given in the training phase.
Target variable `popularity`:
- The optimal configuration appears to be `{’splitter’: ’best’, ’min_samples_split’: 94, ’min_samples_leaf’: 58, ’max_depth’: 5, ’criterion’: ’entropy’}` with an average accuracy of 0.87. However, there is a problem of class imbalance: in fact, tracks with low pop- ularity cover almost the entire dataset, mediums are on the order of hundreds, and highs are a few dozen.
- We can conclude that while the results were promising at first, if we go to consider the weights of the various classes (due to imbalance), the model loses its ability to generalize by a large margin.
### Regression
- **Simple**: The best performing model is the one between `duration_min` and `n_bars`. This confirms what we expected, because the length of the song increases as the number of bars it contains increases. One of the worst performing models is the one between n_bars and tempo. In fact, the number of bars in a song does not significantly influence its tempo, at least not in a linear way.
- **Multiple**: the three best combination target-model are `n_bars`/DecisionTree (with `R2=0.92, MAE=9.15`), `tempo`/DecisionTree and `n_bars`/Lasso.
- **Multivariate**: best performance (`R2=0.49, MAE=4.49`) was achieved by the target `[popularity, danceability, energy]` with model `KNN`.
### Pattern Mining
Our dataset predominantly consists of **non-explicit** tracks with **high volume**, **low speechiness**, a duration range that remains **below 22 minutes**, and a **low number of bars**. Tracks with medium tempo and low speechiness also have a significant presence. The presence of liveness feature among the most frequent patterns also suggests that dataset contains lots of tracks that have a low “live” feel. From association rule extraction ew can see the following information:
- Consequents: in our case it’s mainly related to `energy`.
- Confidence: measures how often the rule has been found to be true. For example, a confidence of 0.72 for the first rule means that in about 72% of the transactions containing `[High_acousticness, Low_valence, Low_liveness, Non-Explicit, Low_speechiness, Low_n_bars]` appear to have a low energy level.
- Lift: ratio of the observed support to that expected if the antecedent and the consequent were independent. In our case, all the higher lifts are around 5, which means that the rules are quite significant.

## Authors

- [@aldomontinaroam](https://github.com/aldomontinaroam)
- [@p-argento](https://github.com/p-argento)
- Lorenzo Lattanzi


## Acknowledgements
- **Python - Anaconda (>3.7)**: Anaconda is the leading open data science platform powered by Python. [Download page](https://www.anaconda.com/distribution/ "https://www.anaconda.com/distribution/")
- Scikit-learn: python library with tools for data mining and data analysis [Documentation page](http://scikit-learn.org/stable/ "http://scikit-learn.org/stable/")
- Pandas: pandas is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language. [Documentation page](http://pandas.pydata.org/ "http://pandas.pydata.org/")
- [Pang-Ning Tan, Michael Steinbach, Vipin Kumar. **Introduction to Data Mining**. Addison Wesley, ISBN 0-321-32136-7, 2006](http://www-users.cs.umn.edu/~kumar/dmbook/index.php "http://www-users.cs.umn.edu/~kumar/dmbook/index.php")
- Header image: AI generated using [Bing](https://www.bing.com/images/create/?ref=hn) and [Adobe Firefly](https://www.adobe.com/it/products/firefly.html).

