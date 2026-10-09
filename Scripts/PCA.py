import pandas as pd

url = "pv_01.csv"

#All Components
df = pd.read_csv(url, names=['hour_of_day','hour_of_day_cos','hour_of_day_sin','month_of_year','month_of_year_cos','month_of_year_sin','season_of_year','season_of_year_cos','season_of_year_sin','sunposition_thetaZ','sunposition_solarAzimuth','sunposition_extraTerr','sunposition_solarHeight','clearsky_diffuse','clearsky_direct','clearsky_global','clearsky_diffuse_agg','clearsky_direct_agg','clearsky_global_agg','Albedo','WindComponentUat0','WindComponentVat0','WindComponentUat100','WindComponentVat100','DewpointTemperatureAt0','TemperatureAt0','PotentialVorticityAt1000','PotentialVorticityAt950','RelativeHumidityAt1000','RelativeHumidityAt950','RelativeHumidityAt0','SnowDensityAt0','SnowDepthAt0','SnowfallPlusStratiformSurfaceAt0','SurfacePressureAt0','SolarRadiationGlobalAt0','SolarRadiationDirectAt0','SolarRadiationDiffuseAt0','TotalCloudCoverAt0','LowerWindSpeed','LowerWindDirection','LowerWindDirectionMath','LowerWindDirectionCos','LowerWindDirectionSin','UpperWindSpeed','UpperWindDirection','UpperWindDirectionMath','UpperWindDirectionCos','UpperWindDirectionSin','power_normed'])

from sklearn.preprocessing import StandardScaler
#All Components of Importance
features = ['hour_of_day','month_of_year','sunposition_thetaZ','sunposition_solarAzimuth','sunposition_extraTerr','sunposition_solarHeight','clearsky_diffuse','clearsky_direct','LowerWindSpeed','LowerWindDirection','UpperWindSpeed','UpperWindDirection']

# Separating out the features
x = df.loc[:, features].values

# Separating out the power_normed
y = df.loc[:,['power_normed']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=12)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2','principal component 3', 'principal component 4','principal component 5', 'principal component 6','principal component 7', 'principal component 8','principal component 9', 'principal component 10','principal component 11', 'principal component 12'])
			 
finalDf = pd.concat([principalDf, df[['power_normed']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_olabel('Principal Component 1', fontsize = 15)
ax.set_plabel('Principal Component 2', fontsize = 15)
ax.set_qlabel('Principal Component 3', fontsize = 15)
ax.set_rlabel('Principal Component 4', fontsize = 15)
ax.set_slabel('Principal Component 5', fontsize = 15)
ax.set_tlabel('Principal Component 6', fontsize = 15)
ax.set_ulabel('Principal Component 7', fontsize = 15)
ax.set_vlabel('Principal Component 8', fontsize = 15)
ax.set_wlabel('Principal Component 9', fontsize = 15)
ax.set_xlabel('Principal Component 10', fontsize = 15)
ax.set_ylabel('Principal Component 11', fontsize = 15)
ax.set_zlabel('Principal Component 12', fontsize = 15)
ax.set_title('12 component PCA', fontsize = 20)
power_normed = ['0', '0.035185185', '0.120987654']

colors = ['r', 'g', 'b']
for power_normed, color in zip(power_norms,colors):
    indicesToKeep = finalDf['power_normed'] == power_normed
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
			   ,finalDf.loc[indicesToKeep, 'principal component 3']
               , finalDf.loc[indicesToKeep, 'principal component 4']
			   ,finalDf.loc[indicesToKeep, 'principal component 5']
               , finalDf.loc[indicesToKeep, 'principal component 6']
			   ,finalDf.loc[indicesToKeep, 'principal component 7']
               , finalDf.loc[indicesToKeep, 'principal component 8']
			   ,finalDf.loc[indicesToKeep, 'principal component 9']
               , finalDf.loc[indicesToKeep, 'principal component 10']
			   ,finalDf.loc[indicesToKeep, 'principal component 11']
               , finalDf.loc[indicesToKeep, 'principal component 12']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

pca.explained_variance_ratio_