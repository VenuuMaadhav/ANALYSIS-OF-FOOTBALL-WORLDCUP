import pandas as pd
wcm=pd.read_csv('C:/Users/HP PC/Desktop/datavisazaition/WorldCupMatches.csv')
wcp=pd.read_csv('C:/Users/HP PC/Desktop/datavisazaition/WorldCupPlayers.csv')
wc=pd.read_csv('C:/Users/HP PC/Desktop/datavisazaition/WorldCups.csv')
wcp = wcp.dropna()
wc = wc.dropna()
wcm = wcm.dropna()
wc = wc.replace('Germany FR', 'Germany')
wcp = wcp.replace('Germany FR', 'Germany')
wcm = wcm.replace('Germany FR', 'Germany')
wcm['Stadium'] = wcm['Stadium'].str.replace('Maracanï¿½ - Estï¿½dio Jornalista Mï¿½rio Filho','Maracanã Stadium')
import numpy as np
import matplotlib.pyplot as plt
winner=wc["Winner"]
runners_up=wc["Runners-Up"]
winner_count=pd.DataFrame.from_dict(winner.value_counts())
runners_up_count=pd.DataFrame.from_dict(runners_up.value_counts())
overall=winner_count.join(runners_up_count, how='outer')
overall=overall.fillna(0)
overall.columns=['WINNER', 'RUNNERS_UP']
overall=overall.astype('int64')
overall=overall.sort_values(by=['WINNER', 'RUNNERS_UP'])
overall.plot(y=['WINNER', 'RUNNERS_UP'], kind="bar", color =['red','blue'], align='center', figsize=(20, 10), grid=True)
plt.xlabel('Countries')
plt.ylabel('Number of times reached final')
plt.title('Number of times any team reaching final')
plt.show()

home_team_goal=wcm.groupby('Year')
away_team_goal=wcm.groupby('Year')
home_vs_away_team_goal=home_team_goal['Home Team Goals', 'Away Team Goals'].sum()
home_vs_away_team_goal.plot(y=['Home Team Goals', 'Away Team Goals'], kind="bar", color=['red', 'blue'], align='center', figsize=(20, 10))
plt.ylabel('Goals')

#****************************
germany_home=wcm.groupby('Home Team Name').get_group('Germany')[['Year', 'Home Team Goals']]
germany_away=wcm.groupby('Away Team Name').get_group('Germany')[['Year', 'Away Team Goals']]
germany_home_goal=germany_home.groupby('Year')['Home Team Goals'].sum()
germany_away_goal=germany_away.groupby('Year')['Away Team Goals'].sum()
germany_home_goal.plot.bar(figsize=(10,5))

germany_away_goal.plot.bar(figsize=(10,5))
import seaborn as sns
att=wcm.sort_values(by='Attendance', ascending=False)[:10]
att=att[['Year', 'Datetime','Stadium', 'City', 'Home Team Name', 'Home Team Goals', 'Away Team Name', 'Attendance']]
att['vs_team']=att["Home Team Name"] + 'vs' + att["Away Team Name"]
plt.figure(figsize=(20,10))
ax=sns.barplot(x=att['Attendance'], y=att['vs_team'], palette='Blues_r', linewidth = 1,edgecolor="k"*len(att))
plt.ylabel('teams')
plt.xlabel('Attendance')
plt.title('Matches with highest number of attendance')
for i,j in enumerate(' stadium : '+att['Stadium']+' , Date :' + att['Datetime']):
    ax.text(1,i,j,fontsize = 15,color='black',weight = 'bold')
plt.show()
avg_std=wcm.groupby(['Stadium', 'City'])['Attendance'].mean().reset_index().sort_values(by='Attendance', ascending=False)[:10]
plt.figure(figsize=(20,10))
ax=sns.barplot(x=avg_std['Attendance'], y=avg_std['Stadium'], palette='Reds_r')
for i,j in enumerate('City:' + avg_std['City'][:10]):
    ax.text(1,i,j,fontsize=15)
plt.title('Stadiums with highest average attendance')
plt.show()

#******************************

# Man of the match Prediction:
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import skew
import matplotlib.pyplot as plt
data = pd.read_csv('C:/Users/rahul/Desktop/FIFA 2018 Statistics.csv')
numerical_features = data.select_dtypes(include = [np.number]).columns
categorical_features = data.select_dtypes(include= [np.object]).columns
print(numerical_features)
print(categorical_features)
var1 = ['Goal Scored', 'On-Target', 'Corners', 'Attempts', 'Free Kicks', 'Yellow Card', 'Red', 'Fouls Committed']
var1.append('Man of the Match')
sns.pairplot(data[var1], hue = 'Man of the Match', palette="husl")
plt.show()

#attempts*****************

attempts=data.groupby('Team')['Attempts'].sum().reset_index().sort_values(by=('Attempts'),ascending=False)
plt.figure(figsize = (15, 12), facecolor = None)
sns.set_style("darkgrid")
sns.set(font_scale=1.5)
plot1 = sns.barplot(x="Team", y="Attempts", data=attempts)
plot1.set_xticklabels(attempts['Team'], rotation=90, ha="center")
plot1.set(xlabel='Teams',ylabel='Total attempts')
plot1.set_title('Total goal attempts by teams')
#Goal Scored by a team**************************
goals_by_team=data.groupby('Team')['Goal Scored'].sum().reset_index().sort_values(by=('Goal Scored'),ascending=False)
plt.figure(figsize = (15,12), facecolor = None)
sns.set_style("darkgrid")
sns.set(font_scale=1.5)
plot1 = sns.barplot(x="Team", y="Goal Scored", data=goals_by_team)
plot1.set_xticklabels(goals_by_team['Team'], rotation=90, ha="center")
plot1.set(xlabel='Teams',ylabel='Total goals scored')
plot1.set_title('Total goals scored by teams')
#Ball Possession***************************
ball_possession=data.groupby('Team')['Ball Possession %'].mean().reset_index().sort_values(by=('Ball Possession %'),ascending=False)
plt.figure(figsize = (12, 10), facecolor = None)
sns.set_style("darkgrid")
sns.set(font_scale=1.5)
plot1 = sns.barplot(x="Team", y="Ball Possession %", data=ball_possession)
plot1.set_xticklabels(ball_possession['Team'], rotation=90, ha="center")
plot1.set(xlabel='Teams',ylabel='Ball possession')
plot1.set_title('Mean ball possession')
#Man of the Match**************************
mom_1={'Man of the Match':{'Yes':1,'No':0}}
data.replace(mom_1,inplace=True)
data['Man of the Match']=data['Man of the Match'].astype(int)
mom=data.groupby('Team')['Man of the Match'].sum().reset_index().sort_values(by=('Man of the Match'),ascending=False)
plt.figure(figsize = (12, 10), facecolor = None)
sns.set_style("darkgrid")
sns.set(font_scale=1.5)
plot1 = sns.barplot(x="Team", y="Man of the Match", data=mom)
plot1.set_xticklabels(mom['Team'], rotation=90, ha="center")
plot1.set(xlabel='Teams',ylabel='Total Man of the Matches')
plot1.set_title('Most Man of the Match awards')
#On-target****************************
group_attempt = data.groupby('Team')['On-Target','Off-Target','Blocked'].sum().reset_index()
group_attempt_sorted = group_attempt.melt('Team', var_name='Target', value_name='Value')
plt.figure(figsize = (16, 10), facecolor = None)
sns.set_style("darkgrid")
sns.set(font_scale=1.5)
plot1 = sns.barplot(x="Team", y="Value", hue="Target", data=group_attempt_sorted)
plot1.set_xticklabels(group_attempt_sorted['Team'], rotation=90, ha="center")
plot1.set(xlabel='Teams',ylabel='Total Attempts')
plot1.set_title('Total On-Target, Off-Target and Blocked attempts by teams')
#saves*****************************
saves=data.groupby('Team')['Saves'].sum().reset_index().sort_values(by=('Saves'),ascending=False)
plt.figure(figsize = (15,12), facecolor = None)
sns.set_style("darkgrid")
plot1 = sns.barplot(x="Team", y="Saves", data=saves)
plot1.set_xticklabels(saves['Team'], rotation=90, ha="center")
plot1.set(xlabel='Teams',ylabel='Total Saves')
plot1.set_title('Most Saves')
#corners*****************************
corners_offsides_freekicks = data.groupby('Team')['Corners','Offsides','Free Kicks'].sum().reset_index()
corners_offsides_freekicks_sort = corners_offsides_freekicks.melt('Team', var_name='Target', value_name='Value')
plt.figure(figsize = (16, 10), facecolor = None)
sns.set_style("darkgrid")
sns.set(font_scale=1.5)
plot1 = sns.barplot(x="Team", y="Value", hue="Target", data=corners_offsides_freekicks_sort)
plot1.set_xticklabels(corners_offsides_freekicks_sort['Team'], rotation=90, ha="center")
plot1.set(xlabel='Teams',ylabel='Totals')
plot1.set_title('Total Corners, free kicks and offsides for teams')
#Goal Scored by Opponent************************
goals_conceded = data.groupby('Opponent')['Goal Scored'].sum().reset_index().sort_values(by=('Goal Scored'), ascending=False)
plt.figure(figsize = (16, 10), facecolor = None)
sns.set_style("darkgrid")
sns.set(font_scale=1.5)
plot1 = sns.barplot(x="Opponent", y="Goal Scored", data=goals_conceded)
plot1.set_xticklabels(goals_conceded['Opponent'], rotation=90, ha="center")
plot1.set(xlabel='Teams',ylabel='Total goals conceded')
plot1.set_title('Total goals conceded')
#YellowCard*******************************
yellow_cards = data.groupby('Team')['Yellow Card'].sum().reset_index().sort_values(by=('Yellow Card'), ascending=False)
plt.figure(figsize = (16, 10), facecolor = None)
sns.set_style("darkgrid")
sns.set(font_scale=1.5)
plot1 = sns.barplot(x="Team", y="Yellow Card", data=yellow_cards)
plot1.set_xticklabels(yellow_cards['Team'], rotation=45, ha="center")
plot1.set(xlabel='Teams',ylabel='Total yellow cards')
plot1.set_title('Total yellow cards')
#skewness*******************************
skew_values = skew(data[numerical_features], nan_policy = 'omit')
pd.concat([pd.DataFrame(list(numerical_features), columns=['Features']),
pd.DataFrame(list(skew_values), columns=['Skewness degree'])], axis = 1)
#missing Values
missing_values = data.isnull().sum().sort_values(ascending = False)
percentage_missing_values = (missing_values/len(data))*100
pd.concat([missing_values, percentage_missing_values], axis = 1, keys= ['Missingvalues', '% Missing']