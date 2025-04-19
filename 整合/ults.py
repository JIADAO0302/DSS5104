from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, RegressorMixin
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_validate, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,mean_absolute_percentage_error
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
# Custom wrapper for statsmodels OLS to be used in scikit-learn pipelines
class OLSWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, add_constant=True):
        self.add_constant = add_constant

    def fit(self, X, y):
        # Add intercept if required (statsmodels does not add it automatically)
        if self.add_constant:
            X = sm.add_constant(X)
        self.model_ = sm.OLS(y, X).fit()
        return self

    def predict(self, X):
        if self.add_constant:
            # Make sure we add the constant if needed during prediction
            # 'has_constant' parameter ensures that a constant column isn’t added twice if it already exists.
            X = sm.add_constant(X, has_constant='add')
        return self.model_.predict(X)

    def summary(self):
        return self.model_.summary()
    

def evaluate_model(pipeline, X_train, y_train, X_test, y_test, scoring, cv):
    # Cross-validation on training set
    cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True)
    
    # Fit the pipeline on the full training set
    pipeline.fit(X_train, y_train)
    # Predict on the test set
    y_pred_test = pipeline.predict(X_test)
    
    # Compute test errors
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    # Convert negative scores to positive values
    train_mae = -np.mean(cv_results['train_MAE'])
    train_rmse = -np.mean(cv_results['train_RMSE'])
    cv_mae = -np.mean(cv_results['test_MAE'])
    cv_rmse = -np.mean(cv_results['test_RMSE'])
    r2   = r2_score(y_test, y_pred_test)
    mape = mean_absolute_percentage_error(y_test, y_pred_test)
    return {
        'Model': pipeline.named_steps['reg'].__class__.__name__,
        'Train MAE': train_mae,
        'Train RMSE': train_rmse,
        'CV MAE': cv_mae,
        'CV RMSE': cv_rmse,
        'Test MAE': test_mae,
        'Test RMSE': test_rmse,
        'Test R2': r2,
        'MAPE': mape,
        'predictions': y_pred_test,
        }
    
def show_summary(pipeline, X_train,y_train):
    """
    Function to display coefficients of the model.
    """
    print('=' * 50 +'\n')
    print("{} Top 5 Features with Coefficients:".format(pipeline.named_steps['reg'].__class__.__name__))
    if pipeline.named_steps['reg'].__class__.__name__ == 'OLSWrapper':
        pipeline.fit(X_train, y_train)
        reg = pipeline.named_steps['reg']
        summary = reg.summary()
        print(summary)
    else:
        pipeline.fit(X_train, y_train)

        # Access the trained lasso regressor from the pipeline
        reg = pipeline.named_steps['reg']

        # Retrieve coefficients and intercept
        coef = reg.coef_
        intercept = reg.intercept_

        # Create a summary table with feature names
        features = X_train.columns
        summary = pd.DataFrame({
            'Feature': ['Intercept'] + list(features),
            'Coefficient': [intercept] + list(coef)
        })
        #将summary逐行打印出来
        for index, row in summary.iterrows():
            print(f"{row['Feature']}: {row['Coefficient']}")
    print('=' * 50 +'\n')

def show_top5_features(pipeline,X_train):
    """
    Function to display top 5 features of the model.
    """
    if pipeline.named_steps['reg'].__class__.__name__ == 'OLSWrapper':
        model = pipeline.named_steps['reg'].model_
        coef = pd.Series(model.params[1:], index=model.model.exog_names[1:])
        top5_features = coef.abs().sort_values(ascending=False).head(5)

    else:
        coef = pd.Series(pipeline.named_steps['reg'].coef_, index=X_train.columns)
        top5_features = coef.abs().sort_values(ascending=False).head(5)

    print(coef.loc[top5_features.index])
    print('=' * 50 +'\n')