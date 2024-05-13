import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# Data preparation: Extracting and translating coefficient labels
file_path = '/Users/vladlobanov/Desktop/ОКРЛЦ и кард.xlsx'
data = pd.read_excel(file_path)
y = data['Геморогический']
X = data[['ИС', 'БА', 'ФП', 'СН', 'СД', 'Атеросклероз', 'Ожирение', 'Возраст', 'Мужчины', 'Женщины']]
X_sm = sm.add_constant(X)
X_train, X_test, y_train, y_test = train_test_split(X_sm, y, test_size=0.3, random_state=42)

# Create and fit the logistic regression model
model = sm.Logit(y_train, X_train)
result = model.fit()

# Extracting coefficients and their standard errors
coef = result.params.drop('const')
errors = result.bse.drop('const')
coef_abs = coef.abs()  # Absolute values of the coefficients
# Data preparation for the histogram
coef_abs_no_gender_age = coef_abs.drop(['Мужчины', 'Женщины', 'Возраст'])

# Correcting the translation process in the labels for the histogram
labels_russian = {
    'ИС': 'Ишемия сердца',
    'БА': 'Бронхиальная астма',
    'ФП': 'Фибрилляция предсердий',
    'СН': 'Сердечная недостаточность',
    'СД': 'Сахарный диабет',
    'Атеросклероз': 'Атеросклероз',
    'Ожирение': 'Ожирение'
}
labels_no_gender_age = [labels_russian.get(label, label) for label in coef_abs_no_gender_age.index]

# Creating the histogram
fig, ax = plt.subplots()
coef_abs_no_gender_age.plot(kind='bar', ax=ax, color='skyblue', label='Абсолютное значение коэффициента')
ax.set_ylabel('Абсолютное значение коэффициента')
ax.set_xlabel('Заболевания')
ax.set_title('Влияние различных заболеваний на риск геморрагического инсульта')
ax.set_xticklabels(labels_no_gender_age, rotation=90)

# Display the plot
plt.legend()
plt.show()






