import pandas as pd

def calculateUnivariantStatsViz(df):
  import seaborn as sns
  from matplotlib import pyplot as plt
  import pandas as pd

  for col in df:
    if pd.api.types.is_numeric_dtype(df[col]):
        f, (ax_box, ax) = plt.subplots(2, sharex = True, gridspec_kw={'height_ratios':(.15,.65)})
        sns.set(style = 'ticks')
        flierprops = dict(marker = 'o', markersize=4, markerfacecolor = 'none', linestyle = 'none', markeredgecolor = 'gray')
        sns.boxplot(x = df[col], ax=ax_box, fliersize=4, width=0.5, linewidth=1, flierprops=flierprops)
        sns.histplot(x = df[col], ax=ax)
        ax_box.set(yticks=[])
        ax_box.set(xticks=[])
        ax_box.set_xlabel('')
        sns.despine(ax=ax)
        sns.despine(ax=ax_box, left=True, bottom=True)
        ax_box.set_title(col, fontsize = 4)

        text = 'Count: ' + str(df[col].count()) + '\n'
        text += 'Unique: ' + str(round(df[col].nunique(), 2)) + '\n'
        text += 'Data Type: ' + str(df[col].dtype) + '\n'
        text += 'Missing: ' + str(round(df[col].isnull().sum(), 2)) + '\n'
        text += 'Mode: ' + str(df[col].mode().values[0]) + '\n'
        text += 'Min: ' + str(round(df[col].min(), 2)) + '\n'
        text += '25%: ' + str(round(df[col].quantile(.25), 2)) + '\n'
        text += 'Median: ' + str(round(df[col].median(), 2)) + '\n'
        text += '75%: ' + str(round(df[col].quantile(.75), 2)) + '\n'
        text += 'Max: ' + str(round(df[col].max(), 2)) + '\n'
        text += 'Std dev: ' + str(round(df[col].std(), 2)) + '\n'
        text += 'Mean: ' + str(round(df[col].mean(), 2)) + '\n'
        text += 'Skew: ' + str(round(df[col].skew(), 2)) + '\n'
        text += 'Kurt: ' + str(round(df[col].kurt(), 2))
        ax.text(0.9, 0.25, text, fontsize=10, transform=plt.gcf().transFigure)
        plt.show()

    else:
      ax_count = sns.countplot(x=col, data=df, order=df[col].value_counts().index, palette=sns.color_palette('RdBu_r', df[col].nunique()))
      sns.despine(ax=ax_count)
      ax_count.set_title(col)
      ax_count.set_xlabel('')
      ax_count.set_ylabel('')

      text = 'Count: ' + str(df[col].count()) + '\n'
      text += 'Unique: ' + str(df[col].nunique()) + '\n'
      text += 'Data Type: ' + str(df[col].dtype) + '\n'
      ax_count.text(0.9, 0.5, text, fontsize=10, transform=plt.gcf().transFigure)
      plt.show()

  stats_df = pd.DataFrame(columns=[ 'Count', 'Unique', 'Type', 'Min', 'Max', '25%', '75%', 'Mean', 'Median', 'Mode', 'Std', 'Skew', 'Kurt', 'NA'])

  for col in df:
    if pd.api.types.is_numeric_dtype(df[col]):

        count = df[col].count()
        unique = round(df[col].nunique(), 2)
        datatype = df[col].dtype
        min = round(df[col].min(), 2)
        max = round(df[col].max(), 2)
        twentyfive = round(df[col].quantile(.25), 2)
        seventyfive = round(df[col].quantile(.75), 2)
        mean = round(df[col].mean(), 2)
        median = round(df[col].median(), 2)
        mode = df[col].mode().values[0]
        std = round(df[col].std(), 2)
        skew = round(df[col].skew(), 2)
        kurt = round(df[col].kurt(), 2)
        NA = 'NO'

        stats_df = stats_df.append({'Feature': col, 'Count':count, 'Unique':unique, 'Type':datatype, 'Min':min, 'Max':max, '25%':twentyfive, '75%':seventyfive, 'Mean':mean, 'Median':median, 'Mode':mode, 'Std':std, 'Skew':skew, 'Kurt':kurt, 'NA':NA}, ignore_index = True)
    else:
      count = df[col].count()
      unique = df[col].nunique()
      datatype = df[col].dtype
      NA = 'YES'

      stats_df = stats_df.append({'Feature': col, 'Count':count, 'Unique':unique, 'Type':datatype, 'Min':None, 'Max':None, '25%':None, '75%':None, 'Mean':None, 'Median':None, 'Mode':None, 'Std':None, 'Skew':None, 'Kurt':None, 'NA':NA}, ignore_index = True)

  return stats_df

def calculateBivariantStatsViz(df, label):
  import pandas as pd

  def calculateTTest(df, feature, label):
    from scipy import stats
    import pandas as pd

    df1 = df[df[feature] == df[feature].unique()[0]]
    df2 = df[df[feature] == df[feature].unique()[1]]

    return stats.ttest_ind(df1[label], df2[label])

  def calculateANOVA(df, feature, label):
    from scipy import stats
    import pandas as pd

    groups = df[feature].unique()
    group_labels = []
    for g in groups:
      group_labels.append(df[df[feature] == g][label])
    return stats.f_oneway(*group_labels)  

  def createBarChart(df, feature, label):
    from matplotlib import pyplot as plt
    import seaborn as sns
    import pandas as pd
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    unique_groups = df[feature].nunique()

    if unique_groups > 2:
      F, p = calculateANOVA(df, feature, label)
      textstr = 'ANOVA' + '\n'
      textstr += 'F-stat:           ' + str(round(F, 2)) + '\n'
      textstr += 'p-value:      ' + str(round(p, 2)) + '\n\n'
      tukey = pairwise_tukeyhsd(endog=df[label],groups=df[feature],alpha=0.05)
      print(tukey)
      ax = sns.barplot(x = df[feature], y = df[label])
      ax.text(1, 0.1, textstr, fontsize=12, transform = plt.gcf().transFigure)
      plt.title(feature + ' and ' + label)
      plt.show()
      return {'Feature': feature, 'Stat': 'F', '+/-':None, 'Effect Size': round(F, 2), 'P-value':round(p, 2)}

    else:
      t, p = calculateTTest(df, feature, label)
      textstr = 'T-Test' + '\n'
      textstr += 'T-stat:           ' + str(round(t, 2)) + '\n'
      textstr += 'p-value:      ' + str(round(p, 2)) + '\n\n'
      ax = sns.barplot(x = df[feature], y = df[label])
      ax.text(1, 0.1, textstr, fontsize=12, transform = plt.gcf().transFigure)
      plt.title(feature + ' and ' + label)
      plt.show()
      return {'Feature': feature, 'Stat': 'T', '+/-':None, 'Effect Size': round(t, 2), 'P-value':round(p, 2)}


  def createScatterPlot(df, feature, label):
    from scipy import stats
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    r, p = stats.pearsonr(df[feature], df[label])
    model = np.polyfit(df[feature], df[label], 1)

    text = 'r - value: ' + str(round(r, 2)) + '\np - value: ' + str(round(p, 2)) 
    text += '\nr2 value: '  + str(round(r * r, 3))
    text += '\ny = ' + str(round(model[0], 2)) + 'x + ' + str(round(model[1], 2))

    sns.set(color_codes = True)
    ax = sns.jointplot(x=df[feature], y=df[label], kind='reg')

    ax.fig.suptitle(feature + ' and ' + label)
    ax.fig.tight_layout()
    ax.fig.subplots_adjust(top=.95)
    ax.fig.text(1, .8, text, fontsize=12, transform=plt.gcf().transFigure)
    plt.show()

    return {'Feature': feature, 'Stat': 'r', '+/-':0.5, 'Effect Size': round(r, 2), 'P-value':round(p, 2)}

  stats_df = pd.DataFrame(columns=['Feature','Stat', '+/-', 'Effect Size', 'P-value'])

  for col in df:
    if pd.api.types.is_numeric_dtype(df[col]) and col != label:
      row = createScatterPlot(df, col, label)
      stats_df = stats_df.append(row, ignore_index=True)
    elif not pd.api.types.is_numeric_dtype(df[col]) and col != label:
      row = createBarChart(df, col, label)
      stats_df = stats_df.append(row, ignore_index=True)

  return stats_df

def assumption1LinearRelationship(df, label):
  def createScatterPlot(df, feature, label):
    from scipy import stats
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    r, p = stats.pearsonr(df[feature], df[label])
    
    if abs(r) < 0.5:
      model = np.polyfit(df[feature], df[label], 1)

      text = 'r - value: ' + str(round(r, 2)) + '\np - value: ' + str(round(p, 2)) 
      text += '\nr2 value: '  + str(round(r * r, 2))
      text += '\ny = ' + str(round(model[0], 2)) + 'x + ' + str(round(model[1], 2))

      sns.set(color_codes = True)
      ax = sns.jointplot(x=df[feature], y=df[label], kind='reg')

      ax.fig.suptitle(feature + ' and ' + label)
      ax.fig.tight_layout()
      ax.fig.subplots_adjust(top=.95)
      ax.fig.text(1, .8, text, fontsize=12, transform=plt.gcf().transFigure)
      plt.show()

      return {'Feature': feature, 'R-Value': round(r, 2)}

  import pandas as pd
  stats_df = pd.DataFrame(columns=['Feature', 'R-Value'])

  for col in df:
    if pd.api.types.is_numeric_dtype(df[col]) and col != label:
      row = createScatterPlot(df, col, label)
      if row != None:
        stats_df = stats_df.append(row, ignore_index=True)

  return stats_df.sort_values(by=['R-Value'], ascending=False)

def assumption2Multicollinearity(df, label):
  from sklearn.linear_model import LinearRegression

  numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
  newdf = df.select_dtypes(include=numerics)

  newdf = newdf.drop([label], axis = 1)
  vif_dict = {}

  for col in newdf:
    y =  newdf[col]
    X = newdf.drop(columns=[col])

    r_squared = LinearRegression().fit(X, y).score(X, y)

    vif = 1/(1 - r_squared)
    vif_dict[col] = round(vif, 4)

  return pd.DataFrame({'VIF': vif_dict}).sort_values(by=['VIF'], ascending=False)

def mlr(df, label):
  from sklearn import preprocessing
  import numpy as np
  import statsmodels.api as sm

  df_dummy3 = df.copy() 

  for col in df_dummy3:
    if not pd.api.types.is_numeric_dtype(df_dummy3[col]): 
      df_dummy3 = df_dummy3.join(pd.get_dummies(df_dummy3[col], prefix=col, drop_first=True)) 

  df_num = df_dummy3.select_dtypes(np.number)

  df_minmax = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(df_num), columns=df_num.columns)

  y = df_minmax[label]
  X = df_minmax.select_dtypes(np.number).assign(const=1)
  X=df_minmax.drop(columns=[label]).assign(const=1)

  model3=sm.OLS(y,X)
  results3=model3.fit()
  
  return results3
  
def assumption3Independence(df, label):
  import numpy as np
  import pandas as pd
  import statsmodels.api as sm
  from statsmodels.stats.stattools import durbin_watson

  results = mlr(df, label)
  dw = round(float(durbin_watson(results.resid)), 3)
  
  if dw <= 2.5 and dw >= 1.5:
    return  "The independence assumption is met.\n" + 'Durbin Watson: ' + str(dw)
  else:
    return "The independence assumption is NOT met." + 'Durbin Watson: ' + str(dw)

def assumption4Homoscedasticity(df, label):
  from statsmodels.compat import lzip
  import statsmodels.stats.api as sms
  import numpy as np
  import pandas as pd
  import statsmodels.api as sm

  model = mlr(df, label)
  
  bp_data = sms.het_breuschpagan(model.resid, model.model.exog)
  names = ['Lagrange multiplier statistic', 'p-value']
  bp_data_dict= dict(lzip(names, bp_data))
  bp_data_dict['Lagrange multiplier statistic'] = round(bp_data_dict['Lagrange multiplier statistic'], 4)
  bp_data_dict['p-value'] = round(bp_data_dict['p-value'], 4)
  bp_df = pd.DataFrame(bp_data_dict, index = ['Breusch-Pagan Values'])

  if bp_df['p-value'][0] > 0.05:
    return (bp_df, 'The Homoscedasticity Assumption IS met.')
  else:
    return (bp_df, 'The Homoscedasticity Assumption is NOT met.') 

def assumption5MultivariateNormality(df, label):
  import numpy as np
  import pandas as pd
  import statsmodels.api as sm
  from statsmodels.stats.stattools import durbin_watson, jarque_bera
  import matplotlib.pyplot as plt
  from scipy import stats
  import statsmodels.stats.api as sms
  from statsmodels.compat import lzip

  df_copy = df.copy()

  for col in df_copy:
    if not pd.api.types.is_numeric_dtype(df_copy[col]):
      df_copy = df_copy.join(pd.get_dummies(df_copy[col], prefix=col, drop_first=True))

  y = df_copy[label]                    
  X = df_copy.select_dtypes(np.number).assign(const=1)
  X = X.drop(columns=[label])

  model = sm.OLS(y, X)
  results = model.fit()

  fig, ax = plt.subplots()

  jb_data = jarque_bera(results.resid)

  names = ['Jarque-Bera Test Statistic', 'P-value']

  jb_data_dict= dict(lzip(names, jb_data))
  jb_data_dict['Jarque-Bera Test Statistic'] = round(jb_data_dict['Jarque-Bera Test Statistic'], 4)
  jb_data_dict['P-value'] = round(jb_data_dict['P-value'], 4)

  jp_df = pd.DataFrame(jb_data_dict, index = ['Jarque-Bera Values'])

  _,(_,_,r) = stats.probplot(results.resid, plot=ax, fit=True)

  if jp_df['P-value'][0] > 0.05:
    return (jp_df, 'The Multivariate Normality Assumption IS met.')
  else:
    return (jp_df, 'The Multivariate Normality Assumption is NOT met.')

def assumptions(df, label):
  print('Assumption #1: Linear Relationship')
  print('Features that don\'t have a linear relationship with ' + str(label) + ":"  )
  print(assumption1LinearRelationship(df, label))
  print('\n\n')

  print('Assumption #2: Multicolinearity')
  print(assumption2Multicollinearity(df, label))
  print('\n\n')

  print('Assumption #3: Independence')
  print(assumption3Independence(df, label))
  print('\n\n')

  print('Assumption #4: Homoscedasticity')
  print(assumption4Homoscedasticity(df, label))
  print('\n\n')

  print('Assumption #5: Multivariate Normality')
  print(assumption5MultivariateNormality(df, label))
  print('\n\n')

def calculateMetrics(df, label):
  import numpy as np
  import pandas as pd
  import statsmodels.api as sm

  df_dummy2 = df.copy()

  for col in df_dummy2:
    if not pd.api.types.is_numeric_dtype(df_dummy2[col]):
      df_dummy2 = df_dummy2.join(pd.get_dummies(df_dummy2[col], prefix=col, drop_first=True))

  y = df_dummy2[label]
  X = df_dummy2.select_dtypes(np.number).assign(const=1)
  X = X.drop(columns=[label])

  model2 = sm.OLS(y,X)
  results2 = model2.fit()

  residuals = np.array(df_dummy2[label]) - np.array(results2.fittedvalues)
  rmse = np.sqrt(sum((residuals**2)) / len(df_dummy2[label]))

  mae = np.mean(abs(residuals))

  return {'R-squared': round(results2.rsquared, 4), 'RMSE' : round(rmse, 4), 'MAE':round(mae, 4), 'Label mean' : round(df_dummy2[label].mean(), 4)}

def calculateMLRandMetrics(df, label):
  print('MLR Results')
  print(mlr(df, label).summary())
  print('\n')

  print('MLR Metrics')
  print(calculateMetrics(df, label))
