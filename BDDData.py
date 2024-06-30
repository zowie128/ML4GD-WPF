import pandas as pd
import numpy as np
import pickle


class BDD_dataset:
    """Class for the BDD Dataset. """

    def __init__(self, data_location):
        self.location_df = pd.read_csv(data_location+"sdwpf_baidukddcup2022_turb_location.csv")
        self.historic_df = pd.read_csv(data_location+"wtbdata_245days.csv")
        self.location_df_original = self.location_df.copy(deep=True)
        max_x = max(self.location_df["x"])
        max_y = max(self.location_df["y"])
        self.location_df["x"] = self.location_df["x"]/max_x
        self.location_df["y"] = self.location_df["y"]/max_y
        self.historic_df = self.historic_df.merge(self.location_df, left_on='TurbID', right_on='TurbID')
        self.sliding_indices = {}

    
    def add_timestep_id(self):
        #Add timestep to df
        self.historic_df['timestep'] = self.historic_df.groupby(['TurbID']).cumcount()
        self.historic_df.loc[:,"timestep"] += 1

    def tag_chaotic(self, replace = False):
        #Filter out weird values
        self.historic_df["zero_value"] =  self.historic_df["Patv"]<0
        self.historic_df["missing_value"] =  self.historic_df["Patv"].isnull()
        self.historic_df["unknown_value"] = (self.historic_df["Patv"]<=0) & (self.historic_df["Wspd"]>2.5)|((self.historic_df["Pab1"]>89)|(self.historic_df["Pab2"]>89)|(self.historic_df["Pab3"]>89))  
        self.historic_df["abnormal_value"] = ((self.historic_df["Ndir"]>720)|(self.historic_df["Ndir"]< -720)) | ((self.historic_df["Wdir"]> 180)|(self.historic_df["Wdir"]< -180))
        self.historic_df["chaotic_value"] = (self.historic_df["zero_value"]) | (self.historic_df["missing_value"]) | (self.historic_df["unknown_value"]) | (self.historic_df["abnormal_value"])
        
        if replace:
            self.historic_df.loc[self.historic_df.chaotic_value, 'Patv'] = np.nan

    def angle_mod(self):
        #Get mod of wind direction
        self.historic_df["Wdir"] = self.historic_df["Wdir"] % 360
        self.historic_df["Ndir"] = self.historic_df["Ndir"] % 360
        self.historic_df["cos_pab1"] = np.cos(self.historic_df["Pab1"] % 360)
        self.historic_df["cos_wdir"] = np.cos(self.historic_df["Wdir"])



    def interpolate_power(self, method="linear"):
        if method == "linear":
            unique_groups = self.historic_df.TurbID.unique()
            for i in range(len(unique_groups)):
                interpolated_series = self.historic_df.loc[self.historic_df["TurbID"]==unique_groups[i],"Patv"].astype(float).interpolate(method='linear')
                self.historic_df.loc[self.historic_df["TurbID"]==unique_groups[i],"Patv"] = interpolated_series
                
            #first_timestep = self.historic_df.loc[self.historic_df["timestep"]==2,"Patv"]
            #self.historic_df.loc[self.historic_df["timestep"]==1,"Patv"] = first_timestep.values

            self.historic_df['Patv'] = self.historic_df.groupby(by = ['TurbID'])['Patv'].bfill()

        if method == "power curve":
            with open("../Patv from Wspd/optimal_params.pickle", 'rb') as handle:
                power_curve_values = pickle.load(handle)
            unique_groups = self.historic_df.TurbID.unique()
            for i in range(len(unique_groups)):
                params_tur = power_curve_values[i]
                self.historic_df["theoretical_patv"] = plf_6(self.historic_df["Wspd"], *(params_tur))
                self.historic_df.loc[self.historic_df['chaotic_value']==True, 'Patv'] = self.historic_df['theoretical_patv']


            self.historic_df['Patv'] = self.historic_df.groupby(by = ['TurbID'])['Patv'].bfill()


    def cap_power_to_zero(self):
        self.historic_df[['Patv']] = self.historic_df[['Patv']].clip(lower = 0)

    def normalize_power(self, min, max, method = "None"):
        self.max_power = np.max(self.historic_df["Patv"])
        self.min_power = np.min(self.historic_df["Patv"])
        self.mean_power = np.mean(self.historic_df["Patv"])
        self.median_power = np.median(self.historic_df["Patv"])
        if (min==0)&(max==1): 
            if method == "MinMaxScaler":
                self.scaler_bias = self.min_power
                self.scaler_weight = self.max_power - self.min_power
                self.historic_df["Patv"] = (self.historic_df["Patv"] - self.scaler_bias) / self.scaler_weight
            else:
                self.historic_df["Patv"] = self.historic_df["Patv"]/self.max_power
        else:
            raise NotImplementedError
        

    def split_df(self, add_features = None):
        #Train 171 days, validation 25 days, test: 49 days
        timesteps_per_day = 24 * 6
        if add_features == None: 
            df = self.historic_df[['TurbID','timestep','Patv']]
            df.loc[:,"TurbID"] -= 1
            df = df.pivot_table(columns='timestep', index='TurbID', values='Patv')
            self.matrix = df.to_numpy()
            train = self.matrix[:,:timesteps_per_day * 171]
            val = self.matrix[:,(timesteps_per_day * 171): timesteps_per_day * (171+25)]
            test = self.matrix[:,(timesteps_per_day * (171+25)):]
        else:
            self.make_tod_circular()
            self.add_farm_status()
            df = self.historic_df[['TurbID','timestep','Patv']+add_features]
            df.loc[:,"TurbID"] -= 1
            df = df.set_index(['TurbID','timestep'])
            self.matrix = np.array(list(df.groupby('TurbID').apply(pd.DataFrame.to_numpy)))
            train = self.matrix[:,:timesteps_per_day * 171,:]
            val = self.matrix[:,(timesteps_per_day * 171): timesteps_per_day * (171+25),:]
            test = self.matrix[:,(timesteps_per_day * (171+25)):,:]

        return train, val, test
    
    def get_observation_forecasting_window(self, time_series_len, observation_steps, forecast_steps,stepsize=1,lag=0):
        sliding_list = []
        for i in range(observation_steps,time_series_len-forecast_steps-lag,stepsize):
            start_index = i-observation_steps+lag
            middle_index = i+lag
            end_index = i + forecast_steps + lag
            #if (middle_index - start_index) == (end_index - middle_index):
            sliding_list.append([start_index,middle_index,end_index])
        self.sliding_indices[str(observation_steps)+","+str(forecast_steps)]=sliding_list