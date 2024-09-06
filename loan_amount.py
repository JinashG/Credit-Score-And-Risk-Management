import csv
import pickle as pk
import random
import string

import joblib
import numpy as np
# import loan_amount_app as lm
import pandas as pd
from flask import Flask, redirect, render_template, request
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

model1 = pk.load(open('model.pkl','rb'))
scaler = pk.load(open('scaler.pkl','rb'))

full_data=pd.read_csv("datasets/train.csv")
# split the data into train and test
def data_split(df, test_size):
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


train_original,test_original = data_split(full_data, 0.2)
train_copy = train_original.copy()
test_copy = test_original.copy()


class OutlierImputer(BaseEstimator, TransformerMixin):
    def __init__(self,feat_with_outliers = ['Income (USD)', 'Loan Amount Request (USD)', 'Current Loan Expenses (USD)', 'Dependents', 'Property Age', 'Property Price']):
        self.feat_with_outliers = feat_with_outliers
    def fit(self,df):
        return self
    def transform(self,df):
        if (set(self.feat_with_outliers).issubset(df.columns)):
            # 25% quantile
            Q1 = df[self.feat_with_outliers].quantile(.25)
            # print(Q1)
            # 75% quantile
            Q3 = df[self.feat_with_outliers].quantile(.75)
            IQR = Q3 - Q1
            # keep the data within 3 IQR
            df = df[~((df[self.feat_with_outliers] < (Q1 - 1.5 * IQR)) | (df[self.feat_with_outliers] > (Q3 + 1.5 * IQR))).any(axis=1)]
            return df
        else:
            print("One or more features are not in the dataframe")
            return df
        
class MissingValueImputer(BaseEstimator, TransformerMixin):
    def __init__(self, mode_imputed_ft = ['Gender','Income Stability','Dependents','Has Active Credit Card','Property Location'], median_imputed_ft = ['Income (USD)', 'Current Loan Expenses (USD)', 'Credit Score', 'Property Age']):
        self.mode_imputed_ft = mode_imputed_ft
        self.median_imputed_ft = median_imputed_ft
    def fit(self,df):
        return self
    def transform(self,df):
        if (set(self.mode_imputed_ft + self.median_imputed_ft).issubset(df.columns)):
            # drop missing values in the target feature
            df.dropna(inplace=True, axis=0, subset=['Loan Sanction Amount (USD)'])
            # impute missing values with mode
            for ft in self.mode_imputed_ft:
                the_mode = df[ft].mode()[0]
                df[ft] = df[ft].fillna(the_mode)
            # impute missing values with median
            for ft in self.median_imputed_ft:
                the_median = df[ft].median()
                df[ft] = df[ft].fillna(the_median)
            return df
        else:
            print("One or more features are not in the dataframe")
            return df
        
class DropUncommonProfession(BaseEstimator, TransformerMixin):
    def __init__(self,profession_list = ['Student','Unemployed','Businessman']):
        self.profession_list = profession_list
    def fit(self,df):
        return self
    def transform(self,df):
        if ('Profession' in df.columns):
            # only keep the professions that are not in the profession_list
            df = df[~df['Profession'].isin(self.profession_list)]
            return df
        else:
            print("Profession feature is not in the dataframe")
            return df
        
class DropFeatures(BaseEstimator,TransformerMixin):
    def __init__(self,feature_to_drop = ['Customer ID','Name','Type of Employment','Property ID']):
        self.feature_to_drop = feature_to_drop
    def fit(self,df):
        return self
    def transform(self,df):
        if (set(self.feature_to_drop).issubset(df.columns)):
            df.drop(self.feature_to_drop,axis=1,inplace=True)
            return df
        else:
            print("One or more features are not in the dataframe")
            return df

class ValueImputer(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_999_val=['Co-Applicant', 'Current Loan Expenses (USD)', 'Loan Sanction Amount (USD)', 'Property Price']):
        self.feat_with_999_val = feat_with_999_val
    def fit(self,df):
        return self
    def transform(self,df):
        if (set(self.feat_with_999_val).issubset(df.columns)):
            for ft in self.feat_with_999_val:
                # replace any occurance of -999.000 with 0
                # df[ft].replace(-999.000,0,inplace=True,regex=True)
                df.loc[:, ft] = df[ft].replace(-999.000, 0, regex=True)
            return df
        else:
            print("One or more features are not in the dataframe")
            return df
        
class MinMaxWithFeatNames(BaseEstimator,TransformerMixin):
    def __init__(self,min_max_scaler_ft = ['Age', 'Income (USD)', 'Loan Amount Request (USD)', 'Current Loan Expenses (USD)', 'Credit Score', 'Property Age', 'Property Price']):
        self.min_max_scaler_ft = min_max_scaler_ft
    def fit(self,df):
        return self
    def transform(self,df):
        if (set(self.min_max_scaler_ft).issubset(df.columns)):
            min_max_enc = MinMaxScaler()
            df[self.min_max_scaler_ft] = min_max_enc.fit_transform(df[self.min_max_scaler_ft])
            return df
        else:
            print("One or more features are not in the dataframe")
            return df
        
class OneHotWithFeatNames(BaseEstimator,TransformerMixin):
    def __init__(self,one_hot_enc_ft = ['Gender', 'Profession', 'Location', 'Expense Type 1', 'Expense Type 2', 'Has Active Credit Card', 'Property Location', 'Income Stability']):
        self.one_hot_enc_ft = one_hot_enc_ft
    def fit(self,df):
        return self
    def transform(self,df):
        if (set(self.one_hot_enc_ft).issubset(df.columns)):
            # function to one hot encode the features in one_hot_enc_ft
            def one_hot_enc(df,one_hot_enc_ft):
                one_hot_enc = OneHotEncoder()
                one_hot_enc.fit(df[one_hot_enc_ft])
                # get the result of the one hot encoding columns names
                feat_names_one_hot_enc = one_hot_enc.get_feature_names_out(one_hot_enc_ft)
                # change the array of the one hot encoding to a dataframe with the column names
                df = pd.DataFrame(one_hot_enc.transform(df[self.one_hot_enc_ft]).toarray(),columns=feat_names_one_hot_enc,index=df.index)
                return df
            # function to concatenat the one hot encoded features with the rest of features that were not encoded
            def concat_with_rest(df,one_hot_enc_df,one_hot_enc_ft):
                # get the rest of the features
                rest_of_features = [ft for ft in df.columns if ft not in one_hot_enc_ft]
                # concatenate the rest of the features with the one hot encoded features
                df_concat = pd.concat([one_hot_enc_df, df[rest_of_features]],axis=1)
                return df_concat
            # one hot encoded dataframe
            one_hot_enc_df = one_hot_enc(df,self.one_hot_enc_ft)
            # returns the concatenated dataframe
            full_df_one_hot_enc = concat_with_rest(df,one_hot_enc_df,self.one_hot_enc_ft)
            return full_df_one_hot_enc
        else:
            print("One or more features are not in the dataframe")
            return df
        
class SkewnessHandler(BaseEstimator, TransformerMixin):
    def __init__(self,col_with_skewness=['Income (USD)','Loan Amount Request (USD)','Current Loan Expenses (USD)','Property Age']):
        self.col_with_skewness = col_with_skewness
    def fit(self,df):
        return self
    def transform(self,df):
        if (set(self.col_with_skewness).issubset(df.columns)):
            # Handle skewness with cubic root transformation
            df[self.col_with_skewness] = np.cbrt(df[self.col_with_skewness])
            return df
        else:
            print('One or more skewed columns are not found')
            return df

def full_pipeline(df):
    pipeline = Pipeline([
        ('drop features', DropFeatures()),
        ('outlier remover', OutlierImputer()),
        ('drop uncommon profession', DropUncommonProfession()),
        ('missing value imputer', MissingValueImputer()),
        ('-999 value imputer', ValueImputer()),
        ('skewness handler', SkewnessHandler()),
        ('min max scaler', MinMaxWithFeatNames()),
        ('one hot encoder', OneHotWithFeatNames())
    ])
    df_pipe_prep = pipeline.fit_transform(df)
    return df_pipe_prep

def predict_amount():
        if request.method == 'POST':
            input_gender = request.form['gender']
            input_age = int(request.form['age'])
            input_income = int(request.form['income'])
            income_stab = request.form['income_stability']
            input_professions = request.form['profession']
            location = request.form['location']
            input_current_loan_amt = int(request.form['current_loan_amt'])
            exp_type_one ="No"
            exp_type_two ="No"
            dependents_count = int(request.form['dependents_count'])
            credit_score = int(request.form['credit_score'])
            loan_default_input_val = 1 if request.form['loan_default_input'] == 'Yes' else 0
            cc_status_input = request.form['cc_status']
            property_age = int(request.form['property_age']) * 365.25
            prop_price = int(request.form['prop_price'])
            # property_type_input = int(request.form['property_type'])
            property_type_input =1
            prop_location = request.form['prop_location']
            co_applicant_val = 1 if request.form['co_applicant_input'] == 'Yes' else 0
            loan_amount_req = int(request.form['loan_amount_req'])
            grad = request.form['grad']
            self_emp = request.form['self_emp']
            Loan_Dur = int(request.form['Loan_Dur'])
            Assets = int(request.form['Assets'])

            profile_to_predict = [
                '',  # customer id2
                '',  # name
                input_gender[:1],
                input_age,
                input_income//83.42,
                income_stab,
                input_professions,
                '',  # type of employment
                location,
                loan_amount_req//83.42,
                input_current_loan_amt//83.42,
                exp_type_one[:1],
                exp_type_two[:1],
                dependents_count,
                credit_score,
                loan_default_input_val,
                cc_status_input,
                0,  # property id
                property_age,
                property_type_input,
                prop_location,
                co_applicant_val,
                prop_price//83.42,
                0  # loan amount sanctioned
            ]
            grad_s = 0 if grad == 'Graduated' else 1
            emp_s = 0 if self_emp == 'No' else 1

            pred_data = pd.DataFrame([[dependents_count, grad_s, emp_s, input_income, loan_amount_req, Loan_Dur, credit_score, Assets]],
                                    columns=['no_of_dependents', 'education', 'self_employed', 'income_annum',
                                            'loan_amount', 'loan_term', 'cibil_score', 'Assets'])
            
            profile_to_predict_df = pd.DataFrame(
                [profile_to_predict], columns=train_copy.columns)
            

            # add the profile to predict as a last row in the train data
            train_copy_with_profile_to_pred = pd.concat(
                [train_copy, profile_to_predict_df], ignore_index=True)
            

            # whole dataset prepared
            profile_to_pred_prep = full_pipeline(
                train_copy_with_profile_to_pred).tail(1).drop(columns=['Loan Sanction Amount (USD)'])

            # Load the model from the local file system
            # model_path = "trained_Random_Forest_Regression.pkl"  # Update this with your actual model file path
            model = joblib.load("trained_Random_Forest_Regression.pkl")
            joblib.dump(model, "trained_Random_Forest_Regression.pkl")

            # Make predictions using the loaded model
            predictions1 = model.predict(profile_to_pred_prep)

            pred_data = scaler.transform(pred_data)
            prediction = model1.predict(pred_data)
            if prediction[0] == 1:
                loan= 'Loan Is Approved'
                return render_template("page4.html",pre=round(predictions1[0] * 83.42, 2),loan=loan)
            else:
                loan ='Loan Is Rejected'
                return render_template("page4.html",loan=loan)

