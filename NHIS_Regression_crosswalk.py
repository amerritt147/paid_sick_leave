#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 13:00:16 2025

@author: alexesmerritt
"""
#IMPORTING PACKAGES
import pandas as pd
import os
os.chdir('/Users/alexesmerritt/Library/CloudStorage/GoogleDrive-akm147@georgetown.edu/.shortcut-targets-by-id/1LAfieGjgpLHN5FpEjgPlbLnLeRr0nfT6/Alexes_Bansal_Lab/')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn import metrics
#from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
import statsmodels.formula.api as smf
import statsmodels.api as sm
import us
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

########################################################################################################################################################################################################
############################################################################################# Variable Set Up ##########################################################################################
########################################################################################################################################################################################################

poly = np.polynomial.polynomial.polyvander

column_name = 'proportions'
county_column_name = 'county'

statefip_to_name = dict([
    ("01", "Alabama"), ("02", "Alaska"), ("04", "Arizona"), ("05", "Arkansas"),
    ("06", "California"), ("08", "Colorado"), ("09", "Connecticut"), ("10", "Delaware"),
    ("11", "District of Columbia"), ("12", "Florida"), ("13", "Georgia"),
    ("15", "Hawaii"), ("16", "Idaho"), ("17", "Illinois"), ("18", "Indiana"),
    ("19", "Iowa"), ("20", "Kansas"), ("21", "Kentucky"), ("22", "Louisiana"),
    ("23", "Maine"), ("24", "Maryland"), ("25", "Massachusetts"), ("26", "Michigan"),
    ("27", "Minnesota"), ("28", "Mississippi"), ("29", "Missouri"), ("30", "Montana"),
    ("31", "Nebraska"), ("32", "Nevada"), ("33", "New Hampshire"), ("34", "New Jersey"),
    ("35", "New Mexico"), ("36", "New York"), ("37", "North Carolina"),
    ("38", "North Dakota"), ("39", "Ohio"), ("40", "Oklahoma"), ("41", "Oregon"),
    ("42", "Pennsylvania"), ("44", "Rhode Island"), ("45", "South Carolina"),
    ("46", "South Dakota"), ("47", "Tennessee"), ("48", "Texas"), ("49", "Utah"),
    ("50", "Vermont"), ("51", "Virginia"), ("53", "Washington"),
    ("54", "West Virginia"), ("55", "Wisconsin"), ("56", "Wyoming")])

edu_mapping = {
    'Grade 1-11': 'High school degree or less',
    '12th grade, no diploma': 'High school degree or less',
    'high school diploma, GED or equivalent': 'High school degree or less',
    'Some college, no degree': 'Some college',
    "Bachelor's degree (Example: BA, AB, BS, BBA)": 'College degree or more',
    "Master's degree (Example: MA, MS, MEng, MEd, MBA)": 'College degree or more',
    "Professional School or Doctoral degree (Example: MD, DDS, DVM, JD, PhD, EdD)": 'College degree or more'}

age_group_mapping = {
    '18–24': '18–34',
    '25–30': '18–34',
    '31–34': '18–34',
    '35–39': '35–49',
    '40–44': '35–49',
    '45–49': '35–49',
    '50–54': '50–64',
    '55–59': '50–64',
    '60–64': '50–64',
    '>65':  '>64',
    '<18':  '<18'}

state_to_division = {
    # New England
    'Connecticut': 'New England', 'Maine': 'New England', 'Massachusetts': 'New England',
    'New Hampshire': 'New England', 'Rhode Island': 'New England', 'Vermont': 'New England',

    # Middle Atlantic
    'New Jersey': 'Middle Atlantic', 'New York': 'Middle Atlantic', 'Pennsylvania': 'Middle Atlantic',

    # East North Central
    'Illinois': 'East North Central', 'Indiana': 'East North Central', 'Michigan': 'East North Central',
    'Ohio': 'East North Central', 'Wisconsin': 'East North Central',

    # West North Central
    'Iowa': 'West North Central', 'Kansas': 'West North Central', 'Minnesota': 'West North Central',
    'Missouri': 'West North Central', 'Nebraska': 'West North Central', 
    'North Dakota': 'West North Central', 'South Dakota': 'West North Central',

    # South Atlantic
    'Delaware': 'South Atlantic', 'Florida': 'South Atlantic', 'Georgia': 'South Atlantic',
    'Maryland': 'South Atlantic', 'North Carolina': 'South Atlantic', 
    'South Carolina': 'South Atlantic', 'Virginia': 'South Atlantic',
    'District of Columbia': 'South Atlantic', 'West Virginia': 'South Atlantic',

    # East South Central
    'Alabama': 'East South Central', 'Kentucky': 'East South Central', 
    'Mississippi': 'East South Central', 'Tennessee': 'East South Central',

    # West South Central
    'Arkansas': 'West South Central', 'Louisiana': 'West South Central',
    'Oklahoma': 'West South Central', 'Texas': 'West South Central',

    # Mountain
    'Arizona': 'Mountain', 'Colorado': 'Mountain', 'Idaho': 'Mountain',
    'Montana': 'Mountain', 'Nevada': 'Mountain', 'New Mexico': 'Mountain',
    'Utah': 'Mountain', 'Wyoming': 'Mountain',

    # Pacific
    'Alaska': 'Pacific', 'California': 'Pacific', 'Hawaii': 'Pacific',
    'Oregon': 'Pacific', 'Washington': 'Pacific'}


race_mapping = {
    'Asian': 'Asian, not Hispanic',
    'Black/African American only': 'Black alone, not Hispanic',
    'Hispanic': 'Hispanic or Latino/a',
    'White': 'White alone, not Hispanic',
    'AIAN': 'Other',
    'Other single and multiple races': 'Other'
                }   


county_pops = pd.read_csv(
        'Paid_Leave/Other_files/populations.csv',
        encoding='latin1',
        on_bad_lines='skip',   # skip broken rows
        low_memory=False    )


county_pops['county'] = county_pops['STATE'] .astype(str).str.zfill(2)+county_pops['COUNTY'].astype(str).str.zfill(3)
county_pops=county_pops.loc[county_pops['YEAR']==3]
county_pops=county_pops[['county','AGE18PLUS_TOT']]

crosswalk=pd.read_csv('Paid_Leave/Other_files/crosswalk_reverse.csv', encoding='latin1')
crosswalk=crosswalk.drop(index=0).rename(columns={'state':'STATEFIP','puma22':'PUMA','afact':'prop_county_in_puma'})
crosswalk['PUMA']= crosswalk['PUMA'].astype(str).str.zfill(5)
crosswalk['STATEFIP']=crosswalk['STATEFIP'].astype(str).str.zfill(2)
crosswalk['county']= crosswalk['county'].astype(str).str.zfill(5)



########################################################################################################################################################################################################
############################################################################################ Model Creation ###########################################################################################
########################################################################################################################################################################################################
def model_selection(yr,bases,family_income_vs_education,race='no',save_line='base_23'):
    
    # Load cleaned NHIS data for the specified year
    df=pd.read_csv('Paid_Leave/cleaned_data/NHIS/cleaned_20'+yr+'.csv')
    
    # Filter to keep only rows where age is not under 18
    df=df.loc[(df['age']>=18)]
    
    # Drop citizenship column as it's not needed for modeling
    #df=df.drop(columns='citizenship')
    
    # Drop any rows with missing values in any column
    df=df.dropna()



    # Define the list of predictor variables
    feature_cols = ['sex','full_time','age',family_income_vs_education,'region','occupation'] #family_income_vs_education allows us easily switch between the two variables for sensitivity analysis
    
    
    # Add race variable if specified
    if race=='yes':
        feature_cols.append('race')
    
    
       
    X = df[feature_cols + ['weight']]  
    y = df['paid_leave']               
    
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, df['weight'], test_size=0.20, random_state=16
    )
    
    # Add target to X_train
    X_train = X_train.copy()
    X_train["target"] = y_train

    # Create formula for logistic regression with categorical variables
    formula = (
    'target ~ C(sex, Treatment(reference="' + bases['sex'] + '")) + '
    'C(region, Treatment(reference="' + bases['region'] + '")) + '
    'C(full_time, Treatment(reference="' + bases['full_time'] + '")) + '
    #'C(emp_hins, Treatment(reference="' + bases['emp_hins'] + '"))+'
    #'C(age, Treatment(reference="' + bases['age'] + '"))+'
    'poly(age, 2) + '
   # 'C(industry, Treatment(reference="' + bases['industry'] + '")) + '
    'C(occupation, Treatment(reference="' + bases['occupation'] + '")) + '
    'C('+family_income_vs_education+', Treatment(reference="' + bases[family_income_vs_education] + '"))+ 1')
    
    
    # Add race to the formula if applicable
    if race=='yes':
        formula=formula+' + C(race,Treatment(reference="'+ bases['race'] + '"))' 
    
    # # Fit logistic regression model using sampling weights
    model = smf.glm(formula=formula,data=X_train,family=sm.families.Binomial(),freq_weights=w_train).fit()
    
    ########################### Plotting Coefficenients ##########################
       
    # Extract model coefficients and confidence intervals
    coefs = model.params
    conf = model.conf_int()
    conf.columns = ['2.5%', '97.5%']
    pvals = model.pvalues
    
    # Build DataFrame
    coef_df = pd.DataFrame({
        'term': coefs.index,
        'coef': coefs.values,
        'lower': conf['2.5%'].values,
        'upper': conf['97.5%'].values,
        'pval': pvals.values
    })
    
    # Remove intercept if present
    coef_df = coef_df[coef_df['term'] != 'Intercept']
    
    # Sort by absolute value of coefficient
    coef_df = coef_df.sort_values(by='coef', key=lambda x: x.abs(), ascending=False).reset_index(drop=True)
    
    # Extract only categorical terms
    categorical_terms = coef_df[coef_df['term'].str.contains(r'^C\(', regex=True)].copy()
    
    # Extract variable name and level from term
    term_parts = categorical_terms['term'].str.extract(r'C\(([^,]+).*?\)\[T\.(.*)\]')
    categorical_terms['variable'] = term_parts[0]
    categorical_terms['level'] = term_parts[1]
    
    # Optional: make them nicer for plotting
    categorical_terms['clean_label'] = categorical_terms['variable'] + ': ' + categorical_terms['level']
    # Define sort orders
    education_order = [
        'Grade 1-11',
        '12th grade, no diploma',
        'high school diploma, GED or equivalent',
        'Some college, no degree',
        "Associate's Degree",
        "Bachelor's degree (Example: BA, AB, BS, BBA)",
        "Master's degree (Example: MA, MS, MEng, MEd, MBA)",
        "Professional School or Doctoral degree (Example: MD, DDS, DVM, JD, PhD, EdD)"
    ]
    race_order=['White', 'Hispanic', 'Black/African American only',
           'Asian', 'AIAN','Other single and multiple races']
    region_order=['Northeast','South','West','Midwest']
    income_order = ['1–2', '2–3', '3–4', '4–5', '5.01+']
    age_order = [ '25–30', '31–34', '35–39', '40–44', '45–49', '50–54', '55–59', '60–64', '>65']
    full_time_order = ['Not Full-time']
    emp_hins_order = ['Employer provided Health Insurance']
    sex_order = ['Female']
    if family_income_vs_education=='family_income':
        sort_maps = {
            'family_income': {v: i for i, v in enumerate(income_order)},
            'age': {v: i for i, v in enumerate(age_order)},
            'full_time': {v: i for i, v in enumerate(full_time_order)},
            'emp_hins': {v: i for i, v in enumerate(emp_hins_order)},
            'sex': {v: i for i, v in enumerate(sex_order)},
            'region': {v: i for i, v in enumerate(region_order)}
        }
    else:
        sort_maps = {
            'education': {v: i for i, v in enumerate(education_order)},
            'age': {v: i for i, v in enumerate(age_order)},
            'full_time': {v: i for i, v in enumerate(full_time_order)},
            'emp_hins': {v: i for i, v in enumerate(emp_hins_order)},
            'sex': {v: i for i, v in enumerate(sex_order)}
        }
    if race=='yes':
        sort_maps['race'] = {v: i for i, v in enumerate(race_order)}
    

    # Assign sort order based on variable-specific logic
    categorical_terms['sort_order'] = categorical_terms.apply(
        lambda row: sort_maps.get(row['variable'], {}).get(row['level'], 999), axis=1
    )
    
    categorical_terms = categorical_terms.sort_values(by=['variable', 'sort_order'])

    # Reverse the order for a cleaner horizontal plot
    categorical_terms = categorical_terms[::-1]
    
    # Create a clean label for plotting
    labels = categorical_terms['clean_label']
    y_pos = range(len(labels))
    
    # Extract values for plotting
    coef = categorical_terms['coef']
    lower = categorical_terms['lower']
    upper = categorical_terms['upper']
    yerr = [coef - lower, upper - coef]
    
    # Plot
    plt.figure(figsize=(10, max(6, 0.4 * len(labels))))
    plt.errorbar(
        x=coef,
        y=y_pos,
        xerr=yerr,
        fmt='o',
        color='steelblue',
        ecolor='gray',
        capsize=3,
        markersize=5
    )
    plt.yticks(y_pos, labels)
    plt.axvline(0, color='black', linestyle='--')
    plt.xlabel('Coefficient')
    plt.title('Categorical Regression Coefficients with 95% CI')
    plt.tight_layout()
    if race=='no':
        plt.savefig('Paid_Leave/Regression_and_outputs/coefficients_'+save_line+'.png')
    else:
        plt.savefig('Paid_Leave/Regression_and_outputs/coefficients_with_race_'+save_line+'.png')
    plt.show()
    
    ########################### Coefficenients DataFrame ##########################
    
    summary_df = pd.DataFrame({
    'term': model.params.index,
    'coef_value': model.params.values,
    'std_err': model.bse.values,
    'z': model.tvalues.values,
    'p_value': model.pvalues.values,
    'conf_low': model.conf_int()[0].values,
    'conf_high': model.conf_int()[1].values
    })
    
    # Extract the level from 'T.' to ']'
    summary_df['coef'] = summary_df['term'].str.extract(r'T\.([^\]]+)')
    
    # Extract the variable name from 'C(' to ','
    summary_df['variable'] = summary_df['term'].str.extract(r'C\(([^,]+),')
    
    # Optional: drop or move 'Intercept' rows if needed
    # summary_df = summary_df[summary_df['term'] != 'Intercept']
    
    # Reorder if you'd like
    summary_df = summary_df[['variable', 'coef', 'coef_value', 'std_err', 'z', 'p_value', 'conf_low', 'conf_high']]
    

    ################################# Identify Threshold ###########################
    test_df = X_test.copy()
    test_df['target'] = y_test
    
    # ⚠️ If using sklearn: y_probs = model.predict_proba(X_test)[:,1]
    # ⚠️ If using statsmodels: y_probs = model.predict(X_test)
    y_probs = model.predict(test_df)
    test_df['predicted_prob'] = y_probs
    
    # 2. Compute threshold metrics for PT & FT
    thresholds = np.arange(0, 1, 0.001)
    results = {}
    optimal_thresholds = {}
    
    for group in ['Part-time', 'Full-time']:
        group_df = test_df[test_df['full_time'] == group]
        y_true_group = group_df['target']
        y_probs_group = group_df['predicted_prob']
    
        df_thr = pd.DataFrame(columns=['Threshold', 'Recall', 'Specificity', 'Difference'])
    
        for thr in thresholds:
            y_pred_custom = (y_probs_group >= thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_custom).ravel()
    
            recall_thr = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity_thr = tn / (tn + fp) if (tn + fp) > 0 else 0
    
            df_thr.loc[len(df_thr)] = [
                thr, recall_thr, specificity_thr, abs(recall_thr - specificity_thr)
            ]
    
        # Save DataFrame + best threshold
        results[group] = df_thr
        optimal_thresholds[group] = df_thr.loc[df_thr['Difference'].idxmin(), 'Threshold']
    
    # 3. Plot Recall & Specificity for both groups
    plt.figure(figsize=(10, 6))
    
    colors = {
        'Part-time': {'recall': 'green', 'specificity': 'green', 'point': 'green'},
        'Full-time': {'recall': 'orange', 'specificity': 'orange', 'point': 'orange'}
    }
    
    for group in ['Part-time', 'Full-time']:
        df_thr = results[group]
        best_thr = optimal_thresholds[group]
        best_row = df_thr.loc[df_thr['Difference'].idxmin()]
    
        # Plot Recall & Specificity curves
        plt.plot(df_thr['Threshold'], df_thr['Recall'],
                 label=f'{group} Recall', color=colors[group]['recall'], linestyle='-.')
        plt.plot(df_thr['Threshold'], df_thr['Specificity'],
                 label=f'{group} Specificity', color=colors[group]['specificity'], linestyle='--')
    
        # Mark optimal threshold
        plt.scatter(best_thr, best_row['Recall'], color=colors[group]['point'],
                    zorder=5, label=f'{group} Optimal Threshold')
        plt.axvline(best_thr, color=colors[group]['point'], linestyle='--', linewidth=1)
    
        # Annotate with threshold value
        plt.annotate(f'{group}\nThr: {best_thr:.2f}',
                     xy=(best_thr, best_row['Recall']),
                     xytext=(best_thr + 0.02, best_row['Recall'] - 0.1),
                     arrowprops=dict(facecolor=colors[group]['point'], arrowstyle='->'))
    
    # Labels and legend
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Recall and Specificity vs. Threshold (Part-time vs Full-time)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if race == 'no':
        plt.savefig('Paid_Leave/Regression_and_outputs/recall_specificity_'+save_line+'.png')
    else:
        plt.savefig('Paid_Leave/Regression_and_outputs/recall_specificity_with_race_'+save_line+'.png')
    
    plt.show()
     
    

   
    ####################### PRINTING STATS ####################################
    
    # # Find threshold where difference between recall and specificity is minimum
    # ideal_threshold=df_thresholding.loc[df_thresholding['Difference'].idxmin(),'Threshold']
    # print('Ideal threshold is: ',str(ideal_threshold))
    
    # y_pred = (y_probs >= ideal_threshold).astype(int)
    
    # score = roc_auc_score(y_test, y_pred)
    # print('ROC AUC SCORE:' +str(score))
    
    # ################################## AGE EFFECT PLOT ##############################

    # # Create a DataFrame with a range of ages
    # age_range = np.arange(18, 66)  # age 18 to 65
    # base_profile = df.iloc[0:1].copy()  # Copy a row to hold everything constant
    
    # # Repeat for each age in range
    # age_effect_df = pd.concat([base_profile]*len(age_range), ignore_index=True)
    # age_effect_df['age'] = age_range  # Replace age column
    
    # # Predict using the model
    # age_effect_df['predicted_prob'] = model.predict(age_effect_df)
    
    # # Plot
    # plt.figure(figsize=(8, 6))
    # plt.plot(age_effect_df['age'], age_effect_df['predicted_prob'], color='green')
    # plt.xlabel('Age')
    # plt.ylabel('Predicted Probability of Paid Leave')
    # plt.title('Effect of Age on Paid Leave Probability')
    # plt.grid(True)
    # plt.tight_layout()
    
    # if race == 'no':
    #     plt.savefig('Paid_Leave/Regression_and_outputs/age_effect_'+save_line+'.png')
    # else:
    #     plt.savefig('Paid_Leave/Regression_and_outputs/age_effect_with_race_'+save_line+'.png')
    # plt.show()
   
    
   
    ########################## CONFUSION MATRIX ################################    
    for group in ['Part-time', 'Full-time']:
        group_df = test_df[test_df['full_time'] == group]
        y_true_group = group_df['target']
        y_probs_group = group_df['predicted_prob']
    
        # Use group-specific threshold
        ideal_threshold = optimal_thresholds[group]
        y_pred = (y_probs_group >= ideal_threshold).astype(int)
    
        # Confusion matrix
        cnf_matrix = metrics.confusion_matrix(y_true_group, y_pred)
        class_names = [0,1]
    
        fig, ax = plt.subplots(figsize=(6,5))
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
    
        # Heatmap
        sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
        ax.xaxis.set_label_position("bottom")
    
        plt.title(f'{group} Confusion Matrix\nThreshold: {ideal_threshold:.3f}', y=1.1)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
    
        # Save file separately for each group
        if race == 'no':
            plt.savefig(f'Paid_Leave/Regression_and_outputs/confusion_matrix_{group}_{save_line}.png')
        else:
            plt.savefig(f'Paid_Leave/Regression_and_outputs/confusion_matrix_{group}_with_race_{save_line}.png')
    
        plt.show()

    return model,optimal_thresholds['Part-time'],optimal_thresholds['Full-time'], coef_df,summary_df



########################################################################################################################################################################################################
############################################################################################ Model Usage ###############################################################################################
########################################################################################################################################################################################################

def model_run(year,logreg,pt_threshold,ft_threshold,family_income_vs_education,race='no',save_line='base_22'):

    #Importing Census Data for those employed
    file_name = 'Paid_Leave/cleaned_data/IPUMS/ipums_employed_'+str(year)+'_cleaned.csv'

    all_results=[]
    assesment_columns=['PUMA', 'PERWT', 'STATEFIP','employment','wfh'] #important columns not used for model
    
    #For race sensistivity, we keep extra variables for use in validation
    if race=='no':
        assesment_columns.append('race')

    if family_income_vs_education=='education':
        assesment_columns.append('family_income')
    else:
        assesment_columns.append('education')
    
    ################# Chunking DataFrame for easier useage #####################
    for ipums_df0 in pd.read_csv(file_name, chunksize=10000):
        ipums=ipums_df0.copy()
        #ipums=ipums.loc[(ipums['age']!='<18')]
 
        ipums=ipums.loc[ (ipums['education']!='Never attended, kindergarten and nursery') ]
        # Keep 'PUMA' for later
        feature_cols=['sex', 'full_time','race','education','family_income', 'emp_hins','age','region','PUMA','PERWT','STATEFIP','industry','occupation','wfh']
        # Select features for prediction
        
     
        #Creating input for dataframe
        X_pums = ipums[feature_cols]
        X_pums=X_pums.dropna()
        results=X_pums
        
        
        if len(X_pums) == 0:
            continue  # skip empty chunks
       
        # Predict
        pums_probs = logreg.predict(X_pums)
        
        
        results['Probability']=pums_probs
        results['Prediction']=np.nan
        results['Prediction'].loc[results['full_time']=='Part-time']=(results['Probability'].loc[results['full_time']=='Part-time'] >= pt_threshold).astype(int)
        results['Prediction'].loc[results['full_time']=='Full-time']=(results['Probability'].loc[results['full_time']=='Full-time'] >= ft_threshold).astype(int)
        results['SAH']=np.nan
        results['SAH'].loc[results['Prediction']==1]=1
        results['SAH'].loc[results['Prediction']==0]=0
        all_results.append(results)
    
   
    final_df = pd.concat(all_results, ignore_index=True)
    
    
    #Importing unemployed dataset 
    unemployed=pd.read_csv('Paid_Leave/cleaned_data/IPUMS/ipums_notemployed_'+str(year)+'_cleaned.csv')
    #All inemployed would not have paid leave, but would be safe at home 
    unemployed['Prediction']=0
    unemployed['SAH']=1
    
    
    final_df=pd.concat([final_df,unemployed], ignore_index=True)
    final_df['PUMA']= final_df['PUMA'].astype(str).str.zfill(5)
    final_df['STATEFIP']=final_df['STATEFIP'].astype(str).str.zfill(2)
    print(final_df.groupby('PUMA')[['SAH','Prediction']].sum())
    final_df.to_csv('Paid_Leave/Regression_and_outputs/model_outcome_'+save_line+'.csv')
    return final_df


########################################################################################################################################################################################################
##################################################################################### PUMA to County Conversion ########################################################################################
########################################################################################################################################################################################################
def puma_crosswalk(df, crosswalk, race='no', save_line='base_23'):
    df = df.copy()
    crosswalk = crosswalk.copy()

    outcomes = {}  # store results for each variable

    for var in ["Prediction"]:
        # Weighted variable at person level
        df["individual_Prediction_pums"] = df[var] * df["PERWT"]

        # Aggregate to PUMA level
        df_sums = (
            df.groupby(["PUMA", "STATEFIP"])[["individual_Prediction_pums", "PERWT"]]
            .sum()
            .reset_index()
        )
        df_sums["individual_Prediction_prop"] = df_sums["individual_Prediction_pums"] / df_sums["PERWT"]

        # Merge with crosswalk
        merged = pd.merge(crosswalk, df_sums, on=["PUMA", "STATEFIP"], how="left")

        # Ensure numerics
        for col in ["individual_Prediction_prop", "prop_county_in_puma", "pop20"]:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

        # Scale to counties
        merged["weighted_Prediction"] = merged["individual_Prediction_prop"] * merged["prop_county_in_puma"]
        merged["PERWT_Prediction"] = merged["PERWT"] * merged["prop_county_in_puma"]

        # Aggregate to county level
        df_outcomes = (
            merged.groupby("county")[["weighted_Prediction", "PERWT_Prediction", "pop20", "prop_county_in_puma"]]
            .sum()
            .reset_index()
        )

        # Merge with county pops
        df_outcomes = df_outcomes.merge(county_pops, on="county", how="left")

        # Final props & counts
        df_outcomes["proportions_Prediction"] = df_outcomes["weighted_Prediction"] / df_outcomes["prop_county_in_puma"]
        df_outcomes["counts_Prediction"] = df_outcomes["proportions_Prediction"] * df_outcomes["pop20"]

        # Store results
        outcomes[var] = (df_outcomes, df_sums, merged)

    # --- Add extra enrichments for Prediction only ---
    df_outcomes_pred, df_sums_pred, merged_pred = outcomes["Prediction"]

    df_outcomes_pred["State"] = (
        df_outcomes_pred["county"].astype(str).str.zfill(5).str[:2].map(statefip_to_name)
    )
    df_outcomes_pred["Census_Division"] = df_outcomes_pred["State"].map(state_to_division)

    # --- SAH ---

    # Merge them together
    df_outcomes_combined = df_outcomes_pred
    df_outcomes_combined.rename(columns={'proportions_Prediction':'proportions'},inplace=True)
    # Save
    df_outcomes_combined.to_csv(
        f"Paid_leave/Regression_and_outputs/Counties_outcomes_with_race_{save_line}.csv", index=False)

    return df_outcomes_combined, df_sums_pred, merged_pred

def mapping(df, column_name, county_column_name,family_income_vs_education,race='yes',save_line='base_22'):
    # Load shapefile
    shapefile = gpd.read_file('Paid_Leave/Other_files/county_shape/cb_2021_us_county_500k.shp')
    shapefile['county_fips'] = shapefile['GEOID'].astype(str).str.zfill(5)

    # Merge your data with the shapefile
    geo_df = shapefile.merge(df, left_on='county_fips', 
                             right_on=county_column_name)

    # Add state name using FIPS prefix
    geo_df['state_fips'] = geo_df['county_fips'].str[:2]
    geo_df['State'] = geo_df['state_fips'].apply(lambda x:
                    us.states.lookup(x).name if us.states.lookup(x) else None)

    # Filter out AK, HI, and territories for continental US
    continental = geo_df[~geo_df['State'].isin(['Alaska', 'Hawaii', 'Puerto Rico',None])]

    import matplotlib.pyplot as plt


    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # Plot the counties
    states_of_interest = ['California','Oregon','Washington','Arizona','Colorado','New Mexico','Michigan','New York','Vermont','Massachusetts','Rhode Island' ,'Connecticut','New Jersey' ,'Maryland','District of Columbia']

    # Load state-level shapefile (assuming you're using the same CRS as counties)
    # You might already have this — otherwise, one source is: gpd.read_file('https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_20m.zip')
    states_gdf = gpd.read_file('Paid_Leave/Other_files/cb_2022_us_state_20m/cb_2022_us_state_20m.shp')  # or URL if remote
    
    # Filter to desired states
    highlight_states = states_gdf[states_gdf['NAME'].isin(states_of_interest)]  # or use FIPS: states_gdf[states_gdf['STATEFP'].isin(['51', '24', '37'])]
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    # Plot counties
    continental.plot(
        column=column_name,
        ax=ax,
        cmap='GnBu',
        linewidth=0.1,
        edgecolor='0.5',norm=norm)
    
    # Plot state borders for selected states
    highlight_states.boundary.plot(ax=ax, color='black', linewidth=1)
    
    # Colorbar setup
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="8%", pad=0.05)
    
    
    sm = plt.cm.ScalarMappable(cmap='GnBu',norm=plt.Normalize(
    vmin=0,
    vmax=1))
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label("Proportion with Paid Leave", fontsize=12)
    ax.axis('off')
    
    
    if column_name =='proportions':
        ax.set_title("County-Level Paid Leave", fontsize=20)
        if race=='no':
            plt.savefig('Paid_Leave/Regression_and_Outputs/paid_leave_output_'
                    +save_line+'.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig('Paid_Leave/Regression_and_Outputs/paid_leave_output_with_race_'
                    +save_line+'.png', dpi=300, bbox_inches='tight')
        ax.axis('off')
    if column_name =='proportions_18+':
        ax.set_title("County-Level Paid Leave 18+", fontsize=20)
        if race=='no':
            plt.savefig('Paid_Leave/Regression_and_Outputs/paid_leave_output_18_'
                    +save_line+'.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig('Paid_Leave/Regression_and_Outputs/paid_leave_output_18_with_race_'
                    +save_line+'.png', dpi=300, bbox_inches='tight')
    if column_name =='proportions_SAH':
        ax.set_title("County-Level Safe at Home", fontsize=20)
        if race=='no':
            plt.savefig('Paid_Leave/Regression_and_Outputs/paid_leave_output_SAH'
                    +save_line+'.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig('Paid_Leave/Regression_and_Outputs/paid_leave_output_SAH_with_race_'
                    +save_line+'.png', dpi=300, bbox_inches='tight')
    if column_name =='proportions_18+_SAH':
        ax.set_title("County-Level Safe at Home 18+", fontsize=20)
        if race=='no':
            plt.savefig('Paid_Leave/Regression_and_Outputs/paid_leave_output_SAH_18'
                    +save_line+'.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig('Paid_Leave/Regression_and_Outputs/paid_leave_output_SAH_18_with_race_'
                    +save_line+'.png', dpi=300, bbox_inches='tight')
   

    plt.show()
    
    
    if race==True:
        df_no_race=pd.read_csv('Paid_leave/Regression_and_outputs/Counties_outcomes_base_23.csv')
        df_no_race['county'] = df_no_race['county'].astype(str).str.zfill(5)
        df_no_race=df_no_race.rename(columns={'proportions':'proportions_no_race'})
        race_diff=continental.merge(df_no_race[['county','proportions_no_race']],how='outer',on='county')
        race_diff['race_diff']=race_diff['proportions_no_race']-race_diff['proportions']

        fig, ax = plt.subplots(1, 1, figsize=(16, 12))

        # Plot the counties
        race_diff.plot(
            column='race_diff',
            ax=ax,
            cmap='PRGn',
            linewidth=0.1)

        ax.set_title("County-Level Difference in Paid Leave w/ Race Covariates", fontsize=20)
        ax.axis('off')

        # Create a divider and append a colorbar axis below the map
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="8%", pad=0.05)  # size = height of colorbar; pad = distance from map

        # Define normalization and colormap
        norm = mpl.colors.TwoSlopeNorm(vmin=-0.15, vcenter=0, vmax=0.15)
        sm = plt.cm.ScalarMappable(cmap='PRGn', norm=norm,)
        sm._A = []  # Dummy array

        # Add the colorbar
        cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
        cbar.set_label("Base Model - Model w/ Race Covariate ", fontsize=12)
        plt.show()


###############################################################################
def run_model_scenario(
    year,
    model_id,
    family_income_vs_education,
    new_base,
    race=False,
    crosswalk=None,
    column_name=None,
    county_column_name=None, save_line='base_22'):
    
    
    # Define bases
    if race==False:
        race='no'
    else:
        race='yes'
        
        
    bases_df=pd.read_csv('Paid_Leave/Other_files/bases.csv')   
    bases=bases_df.set_index('Unnamed: 0')['Lowest Category'].to_dict()
    # Run model selection
    log_reg,pt_threshold,ft_threshold,coef_df,summary_df = model_selection(model_id, bases, family_income_vs_education,save_line=save_line, race=race)
    #return summary_df
    # Run model
    model_outcome = model_run(year, log_reg,pt_threshold,ft_threshold, family_income_vs_education,save_line=save_line, race=race)

    # Set prediction to 0 for non-working individuals
    #model_outcome.loc[model_outcome['employment']!='Employed', 'Prediction'] = 0

    # Calculate individual-level weighted prediction
    model_outcome['individual_prediction_pums'] = model_outcome['Prediction'] * model_outcome['PERWT']

    # Add demographic mappings
    model_outcome['education_2'] = model_outcome['education'].map(edu_mapping)
    model_outcome['age_2'] = model_outcome['age'].map(age_group_mapping)
    model_outcome['race_2'] = model_outcome['race'].map(race_mapping)
    model_outcome['race_gender'] = model_outcome['race_2'] + ' - ' + model_outcome['sex']

    if 'STATEFIP' in model_outcome.columns:
        model_outcome['census_region'] = model_outcome['STATEFIP'].map(statefip_to_name).map(state_to_division)

  
    

    # County-level mapping
    df_map, df_sums, merged_map = puma_crosswalk(model_outcome, crosswalk,save_line=save_line, race=race)
    
    mapping(df_map, column_name, county_column_name,family_income_vs_education, race=race,save_line=save_line)
    #mapping(df_map, 'proportions_18+', county_column_name,family_income_vs_education, race=race,save_line=save_line)
    #mapping(df_map, 'proportions_SAH', county_column_name,family_income_vs_education, race=race,save_line=save_line)
    #mapping(df_map, 'proportions_18+_SAH', county_column_name,family_income_vs_education, race=race,save_line=save_line)
    
    

    return model_outcome, df_map, df_sums, merged_map,coef_df,summary_df


############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################


outcome_base, map_base, sums_base, merged_base,ft_coef,pt_coef   = run_model_scenario(
    year=2023,
    model_id='23',
    family_income_vs_education='education',
    new_base='high school diploma, GED or equivalent',
    race=False,
    crosswalk=crosswalk,
    column_name=column_name,
    county_column_name=county_column_name,save_line='base_23'
)

# outcome_race, demos_race, map_race, sums_race, merged_race,coef_df  = run_model_scenario
#     model_id='23',
#     family_income_vs_education='education',
#     new_base='high school diploma, GED or equivalent',
#     race=True,
#     crosswalk=crosswalk,
#     column_name=column_name,
#     county_column_name=county_column_name)

# outcome_income, demos_income, map_income, sums_income, merged_income = run_model_scenario(
#     year=2022,
#     model_id='22',
#     family_income_vs_education='family_income',
#     new_base='0–1',
#     race=False,
#     crosswalk=crosswalk,
#     column_name=column_name,
#     county_column_name=county_column_name
# )df_sums_employed


# outcome_base_19, demos_base_19, map_base_19, sums_base_19, merged_base_19 = run_model_scenario(
#     year=2019,
#     model_id='19',
#     family_income_vs_education='education',
#     new_base='high school diploma, GED or equivalent',
#     race=False,
#     crosswalk=crosswalk,
#     column_name=column_name,
#     county_column_name=county_column_name
# )