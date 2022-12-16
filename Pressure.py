
import sklearn
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


#---------------------------------#
st.set_page_config(page_title='The Smart Proxy App',
    layout='wide')

#well_df = pd.read_csv("Test_Data.csv")

def build_model(main_df):
    #X = training_df.drop(['GR (API)', 'wellName'], axis=1)
    X = main_df.drop(['Pressure'], axis=1)
    
    Y = main_df['Pressure']
    
    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100-split_size)/100)
    
    st.markdown('**1.5. Data splits**')
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Test set')
    st.info(X_test.shape)
    
    #st.markdown('**1.3. Variable details**:')
    #st.write('X variable')
    #st.info(list(X.columns))
    #st.write('Y variable')
    #st.info(Y.name)
    
    rf = RandomForestRegressor(n_estimators=parameter_n_estimators,
        random_state=parameter_random_state,
        max_features=parameter_max_features,
        criterion=parameter_criterion,
        min_samples_split=parameter_min_samples_split,
       min_samples_leaf=parameter_min_samples_leaf,
        bootstrap=parameter_bootstrap,
        oob_score=parameter_oob_score,
        n_jobs=parameter_n_jobs)
    
    rf.fit(X_train, Y_train)

    st.subheader('2. Model Performance')
    
    st.markdown('**2.1. Training set**')
    Y_pred_train = rf.predict(X_train)
    st.write('Coefficient of determination ($R^2$):')
    st.info( r2_score(Y_train, Y_pred_train) )

    st.write('Error (MSE):')
    st.info( mean_squared_error(Y_train, Y_pred_train) )

    st.markdown('**2.2. Test set**')
    Y_pred_test = rf.predict(X_test)
    st.write('Coefficient of determination ($R^2$):')
    st.info( r2_score(Y_test, Y_pred_test) )

    st.write('Error (MSE):')
    st.info( mean_squared_error(Y_test, Y_pred_test) )

    st.subheader('3. Model Parameters')
    st.write(rf.get_params())

#---------------------------------#
st.write("""
# The Machine Learning App
In this implementation, the *RandomForestRegressor()* function is used in this app for build a regression model using the **Random Forest** algorithm.
Try adjusting the hyperparameters! 
""")

#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    #st.sidebar.markdown("""
    #[Example CSV input file](main_df)
   # """)
    
# Sidebar - Specify parameter settings
with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

with st.sidebar.subheader('2.1. Learning Parameters'):
    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
    parameter_max_features = st.sidebar.select_slider('Max features (max_features)', options=['auto', 'sqrt', 'log2'])
    parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
    parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

with st.sidebar.subheader('2.2. General Parameters'):
    parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
    parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
    parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
    parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
    parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])
#---------------------------------#
# Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df.head(25))
    
    #well_df = pd.read_csv("Test_Data.csv")

    well_df2 = df.dropna(axis=1)

    well_df =well_df2.drop(['Layer_is_Completed', 'Inactive_face_tiers', 'Inactive_line_tiers', 'Inactive_point_tiers',
                'Total_Inactive_tiers'], axis = 1)
  
    #---------------------------------#
     #Getting Tier One Cells                        
    tier1 = []

    list_Ones = ('I', 'J', 'K','N_Porosity', 'N_Permeability', 'N_Initial_Pressure', 'N_(t-1)Pressure', 'N_(t-1)Gas_Saturation', 'N_Grid Top', 'N_Grid Thickness', 'S_Porosity', 'S_Permeability', 'S_Initial_Pressure', 'S_(t-1)Pressure', 'S_(t-1)Gas_Saturation', 'S_Grid Top', 'S_Grid Thickness', 'E_Porosity', 'E_Permeability', 'E_Initial_Pressure', 'E_(t-1)Pressure', 'E_(t-1)Gas_Saturation', 'E_Grid Top', 'E_Grid Thickness', 'W_Porosity', 'W_Permeability', 'W_Initial_Pressure', 'W_(t-1)Pressure', 'W_(t-1)Gas_Saturation', 'W_Grid Top', 'W_Grid Thickness', 'U_Porosity', 'U_Permeability', 'U_Initial_Pressure', 'U_(t-1)Pressure', 'U_(t-1)Gas_Saturation', 'U_Grid Top', 'U_Grid Thickness', 'B_Porosity', 'B_Permeability', 
                 'B_Initial_Pressure', 'B_(t-1)Pressure', 'B_(t-1)Gas_Saturation', 'B_Grid Top', 'B_Grid Thickness', )
    for items in list_Ones:
        tier1.append (items)
    st.markdown( '**1.2. Tier One Wells with Properties**')
    Tier1= well_df[tier1]
    st.write(Tier1.head(20))
    
    #---------------------------------#
    # Get and Display Tier2 Cells
    tier2 = []
    list_Twos = ('I', 'J', 'K','UN_Porosity', 'UN_Permeability', 'UN_Initial_Pressure', 'UN_(t-1)Pressure', 'UN_(t-1)Gas_Saturation', 'UN_Grid Top', 'UN_Grid Thickness', 'US_Porosity', 'US_Permeability', 'US_Initial_Pressure', 'US_(t-1)Pressure', 'US_(t-1)Gas_Saturation', 'US_Grid Top', 'US_Grid Thickness', 'UE_Porosity', 'UE_Permeability', 'UE_Initial_Pressure', 'UE_(t-1)Pressure', 'UE_(t-1)Gas_Saturation', 'UE_Grid Top', 'UE_Grid Thickness', 'UW_Porosity', 'UW_Permeability', 'UW_Initial_Pressure', 'UW_(t-1)Pressure', 'UW_(t-1)Gas_Saturation', 'UW_Grid Top', 'UW_Grid Thickness', 
                 'BN_Porosity', 'BN_Permeability', 'BN_Initial_Pressure', 'BN_(t-1)Pressure', 'BN_(t-1)Gas_Saturation', 'BN_Grid Top', 'BN_Grid Thickness', 'BS_Porosity', 'BS_Permeability', 'BS_Initial_Pressure', 'BS_(t-1)Pressure', 'BS_(t-1)Gas_Saturation', 'BS_Grid Top',
                 'BS_Grid Thickness', 'BE_Porosity', 'BE_Permeability', 'BE_Initial_Pressure', 'BE_(t-1)Pressure', 'BE_(t-1)Gas_Saturation', 'BE_Grid Top', 'BE_Grid Thickness', 'BW_Porosity', 'BW_Permeability', 'BW_Initial_Pressure', 'BW_(t-1)Pressure', 'BW_(t-1)Gas_Saturation', 'BW_Grid Top', 'BW_Grid Thickness', 'NE_Porosity', 'NE_Permeability', 'NE_Initial_Pressure', 'NE_(t-1)Pressure', 'NE_(t-1)Gas_Saturation', 'NE_Grid Top', 'NE_Grid Thickness', 'NW_Porosity', 'NW_Permeability', 'NW_Initial_Pressure', 'NW_(t-1)Pressure',
                 'NW_(t-1)Gas_Saturation', 'NW_Grid Top', 'NW_Grid Thickness', 'SE_Porosity', 'SE_Permeability', 'SE_Initial_Pressure', 'SE_(t-1)Pressure', 'SE_(t-1)Gas_Saturation', 'SE_Grid Top', 'SE_Grid Thickness', 'SW_Porosity', 'SW_Permeability', 'SW_Initial_Pressure', 'SW_(t-1)Pressure', 'SW_(t-1)Gas_Saturation', 'SW_Grid Top', 'SW_Grid Thickness')
    for items in list_Twos:
        tier2.append (items)
    st.markdown( '**1.3. Tier Two Wells with Properties**')
    Tier2= well_df[tier2]
    st.write(Tier2.head(20))
    
    #---------------------------------#
    # Get and Display Tier3 Cells
    
    
    tier3 = []
    list3 = ('I', 'J', 'K','UNE_Porosity', 'UNE_Permeability', 'UNE_Initial_Pressure', 'UNE_(t-1)Pressure', 'UNE_(t-1)Gas_Saturation', 
             'UNE_Grid Top', 'UNE_Grid Thickness', 'USE_Porosity', 'USE_Permeability', 'USE_Initial_Pressure', 'USE_(t-1)Pressure', 
             'USE_(t-1)Gas_Saturation', 'USE_Grid Top', 'USE_Grid Thickness', 'UNW_Porosity', 'UNW_Permeability', 'UNW_Initial_Pressure', 
             'UNW_(t-1)Pressure', 'UNW_(t-1)Gas_Saturation', 'UNW_Grid Top', 'UNW_Grid Thickness', 'USW_Porosity', 'USW_Permeability', 'USW_Initial_Pressure',
             'USW_(t-1)Pressure', 'USW_(t-1)Gas_Saturation', 'USW_Grid Top', 'USW_Grid Thickness', 'BNE_Porosity', 'BNE_Permeability', 'BNE_Initial_Pressure', 'BNE_(t-1)Pressure', 'BNE_(t-1)Gas_Saturation',
             'BNE_Grid Top', 'BNE_Grid Thickness', 'BSE_Porosity', 'BSE_Permeability', 'BSE_Initial_Pressure', 'BSE_(t-1)Pressure', 'BSE_(t-1)Gas_Saturation',
             'BSE_Grid Top', 'BSE_Grid Thickness', 'BNW_Porosity', 'BNW_Permeability', 'BNW_Initial_Pressure', 'BNW_(t-1)Pressure', 'BNW_(t-1)Gas_Saturation', 'BNW_Grid Top', 'BNW_Grid Thickness',
             'BSW_Porosity', 'BSW_Permeability', 'BSW_Initial_Pressure', 'BSW_(t-1)Pressure', 'BSW_(t-1)Gas_Saturation', 'BSW_Grid Top', 'BSW_Grid Thickness')
    
    for items in list3:
        tier3.append (items)
        
    st.markdown( '**1.4. Tier Two Wells with Properties**')
    Tier3= well_df[tier3]
    st.write(Tier3.head(20))
   
   #Build Model 
    build_model(well_df)
    
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        
       
        # Maher dataset
          
        well_df = pd.read_csv("Test_Data.csv")

        well_df2 = well_df.dropna(axis=1)

        main_df =well_df2.drop(['Layer_is_Completed', 'Inactive_face_tiers', 'Inactive_line_tiers', 'Inactive_point_tiers',
                'Total_Inactive_tiers'], axis = 1
                              )
        st.markdown('Maher Illinois dataset is used as the example.')
        st.write(df.head(5))

        build_model(main_df)
