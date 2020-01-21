# ==================== WHAT TO FILL OUT IN THIS FILE ===========================
"""
There are 3 sections that should be filled out in this file

1. Package imports below
2. The PricingModel class
3. The load function

There are also three other sections at the end of the file that can
safely be ignored. These are:

    - Probability calibration. An optional step to ensure
      probabilities predicted by your model are calibrated
      (see docstring)

    - Consistency check to make sure the code that you submit is the
      same as your trained model.

    - A main section that runs this file and produces the the trained model file
      This also checks whether your load_model function works properly
"""
# ========================== 1. PACKAGE IMPORTS ================================
# Include your package imports here
import hashlib
import numpy as np
import pandas as pd
import pickle
import os

from sklearn.calibration import CalibratedClassifierCV

import sklearn as skl
from sklearn.preprocessing import OneHotEncoder
#from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
pd.options.mode.chained_assignment = None
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc, recall_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

import tensorflow as tf
from datetime import datetime
from tensorflow.keras.regularizers import l2

from zipfile import ZipFile


# ========================= 2. THE PRICING MODEL ===============================
class PricingModel():
    """
    This is the PricingModel template. You are allowed to:
    1. Add methods
    2. Add init variables
    3. Fill out code to train the model

    You must ensure that the predict_premium method works!
    """

    def __init__(self,):

        # =============================================================
        # This line ensures that your file (pricing_model.py) and the saved
        # model are consistent and can be used together.
        self._init_consistency_check()  # DO NOT REMOVE
        self.onehot_enc = None
        self.hash_enc = None
        self.scaler = None
        self.y_mean = None
        self.y_std = None
        self.Model_made = None
        self.Model_claim = None


    def _preprocessor(self, X_raw, train):
        """

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : Pandas dataframe
            This is the raw data features excluding claims information 

        Returns
        -------
        X: Pandas DataFrame
            A clean data set that is used for training and prediction.
        """
        # =============================================================
        # YOUR CODE HERE

        df = X_raw.copy()
        drop_index = []
        original_index = df.index.tolist()
        if len(df.shape) == 1:
            df = df.to_frame().transpose()

        categorical_data_onehot  = ["city_district_code","pol_coverage", "pol_pay_freq", "pol_payd", "pol_usage", "drv_drv2", "drv_sex1", "drv_sex2", "vh_fuel", "vh_type", "vh_make"]
        categorical_data_hash = ["pol_insee_code","regional_department_code","canton_code"]
        drop_data = ["id_policy","commune_code"]
        null_data = df.columns[df.isnull().sum()>0].tolist()
        
        #remove anomalies
        if train:
            df.drop(df.loc[df["drv_age_lic2"]>df["drv_age2"]].index,inplace=True)
        
        df.loc[:,"drv_sex2"] = df.loc[:,"drv_sex2"].fillna(value=0)
        if train:
            df.dropna(inplace=True)
        else:
            df = df.fillna(value=0)
        
        #scale continous data
        for col in ["population", "pol_bonus", "pol_sit_duration", "town_mean_altitude", "town_surface_area", "vh_age", "vh_sale_begin", "vh_sale_end", "vh_value", "vh_speed"]:
            df[col] = np.log(df[col]+1e-10)
        
        df.loc[:,"pol_insee_code"] = df.loc[:,"pol_insee_code"].str[:2]
        df.loc[:,"vh"] = df["vh_make"].str.strip() + "_" + df["vh_model"].str.strip()
        bool_df1 = df["vh_make"].value_counts()>3000 #3000
        df.loc[:,"vh_make"].loc[~df["vh_make"].isin(bool_df1[bool_df1].index.tolist())] = "na"
        df.loc[:,categorical_data_onehot] = df[categorical_data_onehot].astype(str)
        df.loc[:,categorical_data_hash] = df[categorical_data_hash].astype(str)

        bool_df2 = df["vh"].value_counts()>1000 #1000
        popular_vh = bool_df2[bool_df2].index.tolist()
        df["vh_onehot"] = df["vh"]
        df["vh_hash"] = df["vh"]
        df["vh_onehot"].loc[~df["vh"].isin(popular_vh)] = "na"
        df["vh_hash"].loc[df["vh"].isin(popular_vh)] = "na"
        categorical_data_onehot.append("vh_onehot")
        categorical_data_hash.append("vh_hash")

        if train:
            self.onehot_enc = OneHotEncoder(sparse=False,handle_unknown="ignore")
            onehot_mat = self.onehot_enc.fit_transform(df[categorical_data_onehot])

            #self.hash_enc = FeatureHasher(n_features=len(categorical_data_hash)*20,input_type="string")
            #hash_mat = self.hash_enc.fit_transform(df[categorical_data_hash].values).toarray()

        else:
            onehot_mat = self.onehot_enc.transform(df[categorical_data_onehot])
            #hash_mat = self.hash_enc.transform(df[categorical_data_hash].values).toarray()


        drop_data += categorical_data_onehot
        drop_data += categorical_data_hash
        drop_data += ["vh","vh_make","vh_model"]

        df.drop(columns=drop_data,inplace=True)
        
        final_index = df.index.tolist()
        drop_index = list(set(original_index) - set(final_index))
        
        #X = np.concatenate((df.values,onehot_mat,hash_mat),axis=1)
        X = np.concatenate((df.values,onehot_mat),axis=1)


        
        return X,drop_index


    def fit(self, X_raw, y_made_claim, y_claims_amount):
        """
        Here you will use the fit function for your pricing model.

        Parameters
        ----------
        X_raw : Pandas DataFrame
            This is the raw data features excluding claims information 
        y_made_claim : Pandas DataFrame
            A one dimensional binary array indicating the presence of accidents
        y_claims_amount: Pandas DataFrame
            A one dimensional array which records the severity of claims (this is
            zero where y_made_claim is zero).

        """

        # YOUR CODE HERE

        # Remember to include a line similar to the one below
        # X_clean = self._preprocessor(X_raw)
        
        # made_metrics = [tf.keras.metrics.AUC(name="auc")]
        # def made_nn_model(metrics, input_shape, lr=0.001):
        #     model = tf.keras.Sequential([
        #         tf.keras.layers.Dense(256,activation="relu",input_shape=(input_shape,),kernel_regularizer=l2(l=0.05)),
        #         tf.keras.layers.Dropout(0.5),
        #         tf.keras.layers.Dense(64,activation="relu",kernel_regularizer=l2(l=0.01)),
        #         tf.keras.layers.Dropout(0.5),
        #         tf.keras.layers.Dense(8,activation="relu",kernel_regularizer=l2(l=0.001)),
        #         tf.keras.layers.Dropout(0.5),
        #         tf.keras.layers.Dense(1,activation="sigmoid")
        #     ])

        #     model.compile(
        #     optimizer=tf.keras.optimizers.Adam(lr=lr),
        #     loss=tf.keras.losses.BinaryCrossentropy(),
        #     metrics=metrics)

        #     return model

        # claim_metrics = [tf.keras.metrics.MeanSquaredError(name="mse")]
        # def claim_nn_model(metrics, input_shape, lr=0.001):
        #     model = tf.keras.Sequential([
        #     tf.keras.layers.Dense(256,activation="relu",input_shape=(input_shape,)),
        #     tf.keras.layers.Dropout(0.5),
        #     tf.keras.layers.Dense(16,activation="relu",input_shape=(input_shape,)),
        #     tf.keras.layers.Dropout(0.5),
        #     tf.keras.layers.Dense(8,activation="relu",input_shape=(input_shape,)),
        #     tf.keras.layers.Dropout(0.5),
        #     tf.keras.layers.Dense(1)
        #     ])
    
        #     model.compile(
        #     optimizer=tf.keras.optimizers.Adam(lr=lr),
        #     loss=tf.keras.losses.MeanSquaredError(),
        #     metrics=metrics)
        #     return model

        
        # X_1, X_1val, y_1, y_1val, y_2, y_2val = train_test_split(X_raw,y_made_claim,y_claims_amount,test_size=0.05)
        # X_1, drop_index = self._preprocessor(X_1, train=True)
        # y_1 = y_1.drop(drop_index).values
        # y_2 = y_2.drop(drop_index).values
    
        # X_1val, drop_index = self._preprocessor(X_1val, train=False)
        # y_1val = y_1val.drop(drop_index).values
        # y_2val = y_2val.drop(drop_index).values
        
        # self.scaler = StandardScaler()
        # X_1 = self.scaler.fit_transform(X_1)
        # X_1val = self.scaler.transform(X_1val)
        
        # #prepare for claim amount
        # X_2 = X_1[y_1==1]
        # y_2 = y_2[y_1==1]
        # X_2val = X_1val[y_1val==1]
        # y_2val = y_1val[y_1val==1]
        
        # self.y_mean = np.mean(y_2)
        # self.y_std = np.std(y_2)
        # y_2 = (y_2 - self.y_mean)/self.y_std
        # y_2val = (y_2val - self.y_mean)/self.y_std

        # #fit made claim
        # logdir = "log" + datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode="min", restore_best_weights=True)
                
        # self.Model_made = made_nn_model(made_metrics, X_1.shape[1], lr=0.0003)
        # History_made = self.Model_made.fit(X_1,y_1,
        #                                    class_weight={0:1,1:10},
        #                                    callbacks=[tensorboard_callback, early_stopping],
        #                                    validation_data = (X_1val, y_1val),
        #                                    epochs=200,
        #                                    batch_size=512)

        # #fit claim amount
        # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, mode="min", restore_best_weights=True)
        # logdir = "log" + datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        
        # self.Model_claim = claim_nn_model(claim_metrics, X_2.shape[1], lr=0.0005)
        # History = self.Model_claim.fit(X_2,y_2,
        #                                callbacks=[tensorboard_callback, early_stopping],
        #                                validation_data=(X_2, y_2),
        #                                epochs=5000,
        #                                batch_size=512)
        
        
        X_1, drop_index = self._preprocessor(X_raw, train=True)
        y_1 = y_made_claim.drop(drop_index).values
        y_2 = y_claims_amount.drop(drop_index).values
    
        scaler = StandardScaler()
        clf_made = RandomForestClassifier(n_estimators=500,class_weight={0:1,1:10},n_jobs=-1,max_depth=10,max_features=33,min_samples_leaf=30)
        self.Model_made = Pipeline([("scale",scaler),("clf",clf_made)])
        self.Model_made.fit(X_1,y_1)
        #self.Model_made = fit_and_calibrate_classifier(self.Model_made, X_1, y_1)
        
        # #prepare for claim amount
        X_2 = X_1[y_1==1]
        y_2 = y_2[y_1==1]
        
        self.y_mean = np.mean(y_2)
        self.y_std = np.std(y_2)
        y_2 = (y_2 - self.y_mean)/self.y_std

        clf_claim = RandomForestRegressor(n_estimators=500,n_jobs=-1,max_depth=10,max_features=30,min_samples_leaf=70)
        self.Model_claim = Pipeline([("scale",scaler),("clf",clf_claim)])
        self.Model_claim.fit(X_2,y_2)
        

        return None


    def predict_premium(self, X_raw):
        """Predicts premiums based on the pricing model.

        Parameters
        ----------
        X_raw : Pandas DataFrame
            This is the raw data features excluding claims information

        Returns
        -------
        Pandas DataFrame
            A one dimensional array of the same length as the input with
            values corresponding to the offered premium prices
        """
        # =============================================================
        # You can include a pricing strategy here
        # For example you could scale all your prices down by a factor

        # YOUR CODE HERE

        # Remember to include a line similar to the one below
        # X_clean = self._preprocessor(X_raw)
        
        X, drop_index = self._preprocessor(X_raw, train=False)
        if len(drop_index) > 0:
            print("Some rows of X_raw contain NAs, the corresponding rows are skipped")
        
        #claim_made = self.Model_made.predict(X)
        claim_made = self.Model_made.predict_proba(X)[:,1]
        #print(sum(claim_made>0.28))
        
        decision_threshold = 0.28 #90% boundary
        
        #claim_made = np.where(claim_made>decision_threshold, 1, 0).reshape(-1)
        claim_made = np.where(claim_made>decision_threshold, 1, 0)
        claim_amount = np.repeat(80,len(claim_made))
        claim_made_idx = np.arange(len(claim_made))[claim_made == 1]
        
        X_claim_amount = X[claim_made_idx,:]
        #pred = self.Model_claim.predict(X_claim_amount).reshape(-1)
        pred = self.Model_claim.predict(X_claim_amount)*self.y_std + self.y_mean
        claim_amount[claim_made == 1] = np.round(pred,4)
        prediction = pd.DataFrame({"claim_amount":claim_amount})
        return prediction


    def save_model(self):
        """
        Saves a trained model to pricing_model.p.
        """

        # =============================================================
        # Default : pickle the trained model. Change this (and the load
        # function, below) only if the library you used does not support
        # pickling.
        # self.Model_made.save("Model_made.h5")
        # self.Model_claim.save("Model_claim.h5")
        # Model_made = self.Model_made
        # Model_claim = self.Model_claim
        # self.Model_made = None
        # self.Model_claim = None
        with open('pricing_model.p', 'wb') as target:
            pickle.dump(self, target)

        # self.Model_made = Model_made
        # self.Model_claim = Model_claim

        # zipObj = ZipFile("model.zip","w")
        # zipObj.write("Model_made.h5")
        # zipObj.write("Model_claim.h5")
        # zipObj.write("pricing_model.p")
        # zipObj.close()

        
    def _init_consistency_check(self):
        """
        INTERNAL METHOD: DO NOT CHANGE.
        Ensures that the saved object is consistent with the file.
        This is done by saving a hash of the module file (pricing_model.py) as
        part of the object.
        For this to work, make sure your source code is named pricing_model.py.
        """
        try:
            with open('pricing_model.py', 'r') as ff:
                code = ff.read()
            m = hashlib.sha256()
            m.update(code.encode())
            self._source_hash = m.hexdigest()
        except Exception as err:
            print('There was an error when saving the consistency check: '
                  '%s (your model will still work).' % err)




# =========================== 3. LOAD FUNCTION =================================
def load_trained_model(filename = 'pricing_model.p'):
    """
    Include code that works in tandem with the PricingModel.save_model() method. 

    This function cannot take any parameters and must return a PricingModel object
    that is trained. 

    By default, this uses pickle, and is compatible with the default implementation
    of PricingGame.save_model. Change this only if your model does not support
    pickling (can happen with some libraries).
    """
    # with ZipFile("model.zip","r") as w:
    #     w.extractall()
    
    with open(filename, 'rb') as model:
        pricingmodel = pickle.load(model)
        
    # pricingmodel.Model_made = tf.keras.models.load_model("Model_made.h5")
    # pricingmodel.Model_claim = tf.keras.models.load_model("Model_claim.h5")
    
    
    return pricingmodel



# ========================= OPTIONAL CALIBRATION ===============================
def fit_and_calibrate_classifier(classifier, X, y):
    """
    Note:  This functions performs probability calibration
    This is an optional tool for you to use, it calibrates the probabilities from 
    your model if need be. 

    For more information see:
    https://scikit-learn.org/stable/modules/calibration.html 
    """
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, train_size=0.85, random_state=0)
    classifier = classifier.fit(X_train, y_train)

    # This line does the calibration for you
    calibrated_classifier = CalibratedClassifierCV(
        classifier, method='sigmoid', cv='prefit').fit(X_cal, y_cal)
    return calibrated_classifier



# ==============================================================================

def check_consistency(trained_model, filename):
    """Returns True if the source file is consistent with the trained model."""
    # First, check that the model supports consistency checking (has _source_hash).
    if not hasattr(trained_model, '_source_hash'):
        return True  # No check was done (so we assume it's all fine).
    trained_source_hash = trained_model._source_hash
    with open(filename, 'r') as ff:
        code = ff.read()
    m = hashlib.sha256()
    m.update(code.encode())
    true_source_hash = m.hexdigest()
    return trained_source_hash == true_source_hash




# ============================ MAIN FUNCTION ================================
# Please do not write any executing code outside of the  __main__ safeguard.
# By default, this code trains your model (using training_data.csv) and saves
# it to pricing_model.p, then checks the consistency of the saved model and
# the pickle file.


if __name__ == '__main__':

    # Load the training data
    training_df = pd.read_csv('training_data.csv')
    y_claims_amount = training_df['claim_amount']
    y_made_claim = training_df['made_claim']
    X_train = training_df.drop(columns=['made_claim', 'claim_amount'])

    # Instantiate the pricing model and fit it
    my_pricing_model = PricingModel()
    my_pricing_model.fit(X_train, y_made_claim, y_claims_amount)

    # Save and load the pricing model
    my_pricing_model.save_model()
    loaded_model = load_trained_model()

    # Generate prices from the loaded model and the instantiated model
    predictions1 = my_pricing_model.predict_premium(X_train)
    predictions2 = loaded_model.predict_premium(X_train)

    # ensure that the prices are the same
    assert np.array_equal(predictions1, predictions2)