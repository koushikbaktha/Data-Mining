#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


cgm_data=pd.read_csv('C:\\Users\\koushikbaktha\\Desktop\\ASU\\CSE572\\Project 1\\Project 1 Files\\CGMData.csv',low_memory=False,usecols=['Date','Time','Sensor Glucose (mg/dL)'])
insulin_data=pd.read_csv('C:\\Users\\koushikbaktha\\Desktop\\ASU\\CSE572\\Project 1\\Project 1 Files\\InsulinData.csv',low_memory=False)


# In[3]:


cgm_data['date_time_stamp']=pd.to_datetime(cgm_data['Date'] + ' ' + cgm_data['Time'])


# In[4]:


date_to_remove=cgm_data[cgm_data['Sensor Glucose (mg/dL)'].isna()]['Date'].unique()


# In[5]:


cgm_data=cgm_data.set_index('Date').drop(index=date_to_remove).reset_index()


# In[6]:


cgm_test=cgm_data.copy()


# In[7]:


cgm_test=cgm_test.set_index(pd.DatetimeIndex(cgm_data['date_time_stamp']))


# In[8]:


insulin_data['date_time_stamp']=pd.to_datetime(insulin_data['Date'] + ' ' + insulin_data['Time'])


# In[9]:


start_of_auto_mode=insulin_data.sort_values(by='date_time_stamp',ascending=True).loc[insulin_data['Alarm']=='AUTO MODE ACTIVE PLGM OFF'].iloc[0]['date_time_stamp']


# In[10]:


auto_mode_data_df=cgm_data.sort_values(by='date_time_stamp',ascending=True).loc[cgm_data['date_time_stamp']>=start_of_auto_mode]


# In[11]:


manual_mode_data_df=cgm_data.sort_values(by='date_time_stamp',ascending=True).loc[cgm_data['date_time_stamp']<start_of_auto_mode]


# In[12]:


auto_mode_data_df_date_index=auto_mode_data_df.copy()


# In[13]:


# auto_mode_data_df_date_index=auto_mode_data_df_date_index.replace('',np.nan)
# auto_mode_data_df_date_index=auto_mode_data_df_date_index.replace('NaN',np.nan)


# In[14]:


auto_mode_data_df_date_index=auto_mode_data_df_date_index.set_index('date_time_stamp')


# In[15]:


list1=auto_mode_data_df_date_index.groupby('Date')['Sensor Glucose (mg/dL)'].count().where(lambda x:x>0.8*288).dropna().index.tolist()
# 


# In[16]:


auto_mode_data_df_date_index=auto_mode_data_df_date_index.loc[auto_mode_data_df_date_index['Date'].isin(list1)]


# ### % in Hyperglycemia (> 180 mg/dL) - wholeday, daytime, overnight

# In[17]:


percent_time_in_hyperglycemia_wholeday_automode=(auto_mode_data_df_date_index.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_data_df_date_index['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[18]:


percent_time_in_hyperglycemia_daytime_automode=(auto_mode_data_df_date_index.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_data_df_date_index['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[19]:


percent_time_in_hyperglycemia_overnight_automode=(auto_mode_data_df_date_index.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_data_df_date_index['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# ### % in Hyperglycemia critical (> 250 mg/dL) - wholeday, daytime, overnight

# In[20]:


percent_time_in_hyperglycemia_critical_wholeday_automode=(auto_mode_data_df_date_index.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_data_df_date_index['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[21]:


percent_time_in_hyperglycemia_critical_daytime_automode=(auto_mode_data_df_date_index.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_data_df_date_index['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[22]:


percent_time_in_hyperglycemia_critical_overnight_automode=(auto_mode_data_df_date_index.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_data_df_date_index['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# ### %  in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL) - wholeday, daytime, overnight

# In[23]:


percent_time_in_range_wholeday_automode=(auto_mode_data_df_date_index.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(auto_mode_data_df_date_index['Sensor Glucose (mg/dL)']>=70) & (auto_mode_data_df_date_index['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[24]:


percent_time_in_range_daytime_automode=(auto_mode_data_df_date_index.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(auto_mode_data_df_date_index['Sensor Glucose (mg/dL)']>=70) & (auto_mode_data_df_date_index['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[25]:


percent_time_in_range_overnight_automode=(auto_mode_data_df_date_index.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(auto_mode_data_df_date_index['Sensor Glucose (mg/dL)']>=70) & (auto_mode_data_df_date_index['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# ### %  in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL) - wholeday, daytime, overnight

# In[26]:


percent_time_in_range_sec_wholeday_automode=(auto_mode_data_df_date_index.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(auto_mode_data_df_date_index['Sensor Glucose (mg/dL)']>=70) & (auto_mode_data_df_date_index['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[27]:


percent_time_in_range_sec_daytime_automode=(auto_mode_data_df_date_index.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(auto_mode_data_df_date_index['Sensor Glucose (mg/dL)']>=70) & (auto_mode_data_df_date_index['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[28]:


percent_time_in_range_sec_overnight_automode=(auto_mode_data_df_date_index.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(auto_mode_data_df_date_index['Sensor Glucose (mg/dL)']>=70) & (auto_mode_data_df_date_index['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# ### % in hypoglycemia level 1 (CGM < 70 mg/dL) - wholeday, daytime, overnight

# In[29]:


percent_time_in_hypoglycemia_lv1_wholeday_automode=(auto_mode_data_df_date_index.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_data_df_date_index['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[30]:


percent_time_in_hypoglycemia_lv1_daytime_automode=(auto_mode_data_df_date_index.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_data_df_date_index['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[31]:


percent_time_in_hypoglycemia_lv1_overnight_automode=(auto_mode_data_df_date_index.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_data_df_date_index['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# ### % in hypoglycemia level 2 (CGM < 54 mg/dL) - wholeday, daytime, overnight

# In[32]:


percent_time_in_hypoglycemia_lv2_wholeday_automode=(auto_mode_data_df_date_index.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_data_df_date_index['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[33]:


percent_time_in_hypoglycemia_lv2_daytime_automode=(auto_mode_data_df_date_index.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_data_df_date_index['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[34]:


percent_time_in_hypoglycemia_lv2_overnight_automode=(auto_mode_data_df_date_index.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[auto_mode_data_df_date_index['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[35]:


manual_mode_data_df_index=manual_mode_data_df.copy()
manual_mode_data_df_index=manual_mode_data_df_index.set_index('date_time_stamp')


# In[36]:


# manual_mode_data_df_index=manual_mode_data_df_index.interpolate(columns='Sensor Glucose (mg/dL)')


# In[37]:


# manual_mode_data_df_index=manual_mode_data_df_index.replace('',np.nan)
# manual_mode_data_df_index=manual_mode_data_df_index.replace('NaN',np.nan)


# In[38]:


list2=manual_mode_data_df_index.groupby('Date')['Sensor Glucose (mg/dL)'].count().where(lambda x:x>0.8*288).dropna().index.tolist()
# 


# In[39]:


manual_mode_data_df_index=manual_mode_data_df_index.loc[manual_mode_data_df_index['Date'].isin(list2)]


# ### % in Hyperglycemia (> 180 mg/dL) - wholeday, daytime, overnight

# In[40]:


percent_time_in_hyperglycemia_wholeday_manual=(manual_mode_data_df_index.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_data_df_index['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[41]:


percent_time_in_hyperglycemia_daytime_manual=(manual_mode_data_df_index.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_data_df_index['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[42]:


percent_time_in_hyperglycemia_overnight_manual=(manual_mode_data_df_index.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_data_df_index['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# ### % in Hyperglycemia critical (> 250 mg/dL) - wholeday, daytime, overnight

# In[43]:


percent_time_in_hyperglycemia_critical_wholeday_manual=(manual_mode_data_df_index.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_data_df_index['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[44]:


percent_time_in_hyperglycemia_critical_daytime_manual=(manual_mode_data_df_index.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_data_df_index['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[45]:


percent_time_in_hyperglycemia_critical_overnight_manual=(manual_mode_data_df_index.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_data_df_index['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# ### %  in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL) - wholeday, daytime, overnight

# In[46]:


percent_time_in_range_wholeday_manual=(manual_mode_data_df_index.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(manual_mode_data_df_index['Sensor Glucose (mg/dL)']>=70) & (manual_mode_data_df_index['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[47]:


percent_time_in_range_daytime_manual=(manual_mode_data_df_index.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(manual_mode_data_df_index['Sensor Glucose (mg/dL)']>=70) & (manual_mode_data_df_index['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[48]:


percent_time_in_range_overnight_manual=(manual_mode_data_df_index.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(manual_mode_data_df_index['Sensor Glucose (mg/dL)']>=70) & (manual_mode_data_df_index['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# ### %  in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL) - wholeday, daytime, overnight

# In[49]:


percent_time_in_range_sec_wholeday_manual=(manual_mode_data_df_index.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(manual_mode_data_df_index['Sensor Glucose (mg/dL)']>=70) & (manual_mode_data_df_index['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[50]:


percent_time_in_range_sec_daytime_manual=(manual_mode_data_df_index.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(manual_mode_data_df_index['Sensor Glucose (mg/dL)']>=70) & (manual_mode_data_df_index['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[51]:


percent_time_in_range_sec_overnight_manual=(manual_mode_data_df_index.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(manual_mode_data_df_index['Sensor Glucose (mg/dL)']>=70) & (manual_mode_data_df_index['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# ### % in hypoglycemia level 1 (CGM < 70 mg/dL) - wholeday, daytime, overnight

# In[52]:


percent_time_in_hypoglycemia_lv1_wholeday_manual=(manual_mode_data_df_index.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_data_df_index['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[53]:


percent_time_in_hypoglycemia_lv1_daytime_manual=(manual_mode_data_df_index.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_data_df_index['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[54]:


percent_time_in_hypoglycemia_lv1_overnight_manual=(manual_mode_data_df_index.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_data_df_index['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# ### % in hypoglycemia level 2 (CGM < 54 mg/dL) - wholeday, daytime, overnight

# In[55]:


percent_time_in_hypoglycemia_lv2_wholeday_manual=(manual_mode_data_df_index.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_data_df_index['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[56]:


percent_time_in_hypoglycemia_lv2_daytime_manual=(manual_mode_data_df_index.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_data_df_index['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# In[57]:


percent_time_in_hypoglycemia_lv2_overnight_manual=(manual_mode_data_df_index.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manual_mode_data_df_index['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# ### convert to a dataframe with all values in auto mode and manual mode

# In[58]:


results_df = pd.DataFrame({'percent_time_in_hyperglycemia_overnight':[ percent_time_in_hyperglycemia_overnight_manual.mean(axis=0),percent_time_in_hyperglycemia_overnight_automode.mean(axis=0)],


'percent_time_in_hyperglycemia_critical_overnight':[ percent_time_in_hyperglycemia_critical_overnight_manual.mean(axis=0),percent_time_in_hyperglycemia_critical_overnight_automode.mean(axis=0)],


'percent_time_in_range_overnight':[ percent_time_in_range_overnight_manual.mean(axis=0),percent_time_in_range_overnight_automode.mean(axis=0)],


'percent_time_in_range_sec_overnight':[ percent_time_in_range_sec_overnight_manual.mean(axis=0),percent_time_in_range_sec_overnight_automode.mean(axis=0)],


'percent_time_in_hypoglycemia_lv1_overnight':[ percent_time_in_hypoglycemia_lv1_overnight_manual.mean(axis=0),percent_time_in_hypoglycemia_lv1_overnight_automode.mean(axis=0)],


'percent_time_in_hypoglycemia_lv2_overnight':[ np.nan_to_num(percent_time_in_hypoglycemia_lv2_overnight_manual.mean(axis=0)),percent_time_in_hypoglycemia_lv2_overnight_automode.mean(axis=0)],
                           'percent_time_in_hyperglycemia_daytime':[ percent_time_in_hyperglycemia_daytime_manual.mean(axis=0),percent_time_in_hyperglycemia_daytime_automode.mean(axis=0)],
                           'percent_time_in_hyperglycemia_critical_daytime':[ percent_time_in_hyperglycemia_critical_daytime_manual.mean(axis=0),percent_time_in_hyperglycemia_critical_daytime_automode.mean(axis=0)],
                           'percent_time_in_range_daytime':[ percent_time_in_range_daytime_manual.mean(axis=0),percent_time_in_range_daytime_automode.mean(axis=0)],
                           'percent_time_in_range_sec_daytime':[ percent_time_in_range_sec_daytime_manual.mean(axis=0),percent_time_in_range_sec_daytime_automode.mean(axis=0)],
                           'percent_time_in_hypoglycemia_lv1_daytime':[ percent_time_in_hypoglycemia_lv1_daytime_manual.mean(axis=0),percent_time_in_hypoglycemia_lv1_daytime_automode.mean(axis=0)],
                           'percent_time_in_hypoglycemia_lv2_daytime':[ percent_time_in_hypoglycemia_lv2_daytime_manual.mean(axis=0),percent_time_in_hypoglycemia_lv2_daytime_automode.mean(axis=0)],

                           
                           'percent_time_in_hyperglycemia_wholeday':[ percent_time_in_hyperglycemia_wholeday_manual.mean(axis=0),percent_time_in_hyperglycemia_wholeday_automode.mean(axis=0)],
                           'percent_time_in_hyperglycemia_critical_wholeday':[ percent_time_in_hyperglycemia_critical_wholeday_manual.mean(axis=0),percent_time_in_hyperglycemia_critical_wholeday_automode.mean(axis=0)],
                           'percent_time_in_range_wholeday':[ percent_time_in_range_wholeday_manual.mean(axis=0),percent_time_in_range_wholeday_automode.mean(axis=0)],
                           'percent_time_in_range_sec_wholeday':[ percent_time_in_range_sec_wholeday_manual.mean(axis=0),percent_time_in_range_sec_wholeday_automode.mean(axis=0)],
                           'percent_time_in_hypoglycemia_lv1_wholeday':[ percent_time_in_hypoglycemia_lv1_wholeday_manual.mean(axis=0),percent_time_in_hypoglycemia_lv1_wholeday_automode.mean(axis=0)],
                           'percent_time_in_hypoglycemia_lv2_wholeday':[ percent_time_in_hypoglycemia_lv2_wholeday_manual.mean(axis=0),percent_time_in_hypoglycemia_lv2_wholeday_automode.mean(axis=0)]
                    
                          
},
                          index=['manual_mode','auto_mode'])


# In[59]:


results_df.to_csv('C:\\Users\\koushikbaktha\\Desktop\\ASU\\CSE572\\Project 1\\Project 1 Files\\Results.csv',header=False,index=False)


# In[ ]:




