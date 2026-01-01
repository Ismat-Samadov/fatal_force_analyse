import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for professional business charts
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Create output directory
output_dir = Path("charts")
output_dir.mkdir(exist_ok=True)

# Load datasets
print("Loading datasets...")
deaths = pd.read_csv("data/Deaths_by_Police_US.csv", encoding='latin-1')
income = pd.read_csv("data/Median_Household_Income_2015.csv", encoding='latin-1')
education = pd.read_csv("data/Pct_Over_25_Completed_High_School.csv", encoding='latin-1')
poverty = pd.read_csv("data/Pct_People_Below_Poverty_Level.csv", encoding='latin-1')
race_share = pd.read_csv("data/Share_of_Race_By_City.csv", encoding='latin-1')

# Clean column names
income.columns = income.columns.str.strip()
education.columns = education.columns.str.strip()
poverty.columns = poverty.columns.str.strip()
race_share.columns = race_share.columns.str.strip()

# Data preprocessing
deaths['date'] = pd.to_datetime(deaths['date'], format='%d/%m/%y')
deaths['year'] = deaths['date'].dt.year
deaths['month'] = deaths['date'].dt.month

print(f"Analyzing {len(deaths)} incidents...")

# ============================================================================
# CHART 1: Temporal Trends - Monthly Incident Volume
# ============================================================================
print("Generating Chart 1: Temporal trends...")
fig, ax = plt.subplots(figsize=(14, 6))
monthly_counts = deaths.groupby(deaths['date'].dt.to_period('M')).size()
monthly_counts.index = monthly_counts.index.to_timestamp()
ax.plot(monthly_counts.index, monthly_counts.values, linewidth=2.5, color='#d62728')
ax.fill_between(monthly_counts.index, monthly_counts.values, alpha=0.3, color='#d62728')
ax.set_title('Fatal Police Encounters: Monthly Trend Analysis', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Time Period', fontsize=12)
ax.set_ylabel('Number of Incidents', fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir / "01_temporal_trends.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# CHART 2: Geographic Distribution - Top States
# ============================================================================
print("Generating Chart 2: Geographic distribution...")
fig, ax = plt.subplots(figsize=(14, 8))
state_counts = deaths['state'].value_counts().head(15)
colors = sns.color_palette("Reds_r", len(state_counts))
bars = ax.barh(range(len(state_counts)), state_counts.values, color=colors)
ax.set_yticks(range(len(state_counts)))
ax.set_yticklabels(state_counts.index)
ax.set_xlabel('Number of Fatal Incidents', fontsize=12)
ax.set_title('Geographic Concentration: States with Highest Incident Rates', fontsize=16, fontweight='bold', pad=20)
ax.invert_yaxis()
for i, (idx, val) in enumerate(state_counts.items()):
    ax.text(val + 5, i, f'{val:,}', va='center', fontsize=10, fontweight='bold')
ax.grid(True, axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(output_dir / "02_geographic_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# CHART 3: Demographic Analysis - Race Distribution
# ============================================================================
print("Generating Chart 3: Race distribution...")
fig, ax = plt.subplots(figsize=(12, 7))
race_counts = deaths['race'].value_counts()
race_labels = {
    'W': 'White',
    'B': 'Black',
    'H': 'Hispanic',
    'A': 'Asian',
    'N': 'Native American',
    'O': 'Other'
}
race_counts.index = race_counts.index.map(lambda x: race_labels.get(x, 'Unknown'))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
bars = ax.bar(range(len(race_counts)), race_counts.values, color=colors[:len(race_counts)])
ax.set_xticks(range(len(race_counts)))
ax.set_xticklabels(race_counts.index, rotation=45, ha='right')
ax.set_ylabel('Number of Incidents', fontsize=12)
ax.set_title('Demographic Profile: Racial Distribution of Fatal Encounters', fontsize=16, fontweight='bold', pad=20)
for i, (idx, val) in enumerate(race_counts.items()):
    ax.text(i, val + 10, f'{val:,}\n({val/len(deaths)*100:.1f}%)', ha='center', fontsize=10, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(output_dir / "03_race_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# CHART 4: Age Distribution Analysis
# ============================================================================
print("Generating Chart 4: Age distribution...")
fig, ax = plt.subplots(figsize=(14, 6))
age_data = deaths['age'].dropna()
bins = [0, 18, 25, 35, 45, 55, 65, 100]
labels = ['Under 18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
age_groups = pd.cut(age_data, bins=bins, labels=labels)
age_counts = age_groups.value_counts().sort_index()
colors = sns.color_palette("YlOrRd", len(age_counts))
bars = ax.bar(range(len(age_counts)), age_counts.values, color=colors)
ax.set_xticks(range(len(age_counts)))
ax.set_xticklabels(age_counts.index, rotation=0)
ax.set_ylabel('Number of Incidents', fontsize=12)
ax.set_xlabel('Age Group', fontsize=12)
ax.set_title('Age Distribution: Risk Profile Across Age Groups', fontsize=16, fontweight='bold', pad=20)
for i, (idx, val) in enumerate(age_counts.items()):
    ax.text(i, val + 5, f'{val:,}', ha='center', fontsize=10, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(output_dir / "04_age_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# CHART 5: Armed Status Analysis
# ============================================================================
print("Generating Chart 5: Armed status...")
fig, ax = plt.subplots(figsize=(14, 8))
armed_counts = deaths['armed'].value_counts().head(12)
colors = sns.color_palette("Oranges_r", len(armed_counts))
bars = ax.barh(range(len(armed_counts)), armed_counts.values, color=colors)
ax.set_yticks(range(len(armed_counts)))
ax.set_yticklabels(armed_counts.index)
ax.set_xlabel('Number of Incidents', fontsize=12)
ax.set_title('Weapon Presence: Armed Status at Time of Incident', fontsize=16, fontweight='bold', pad=20)
ax.invert_yaxis()
for i, (idx, val) in enumerate(armed_counts.items()):
    ax.text(val + 5, i, f'{val:,} ({val/len(deaths)*100:.1f}%)', va='center', fontsize=10, fontweight='bold')
ax.grid(True, axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(output_dir / "05_armed_status.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# CHART 6: Mental Illness Indicator
# ============================================================================
print("Generating Chart 6: Mental illness factor...")
fig, ax = plt.subplots(figsize=(10, 6))
mental_counts = deaths['signs_of_mental_illness'].value_counts()
colors = ['#d62728', '#2ca02c']
bars = ax.bar(range(len(mental_counts)), mental_counts.values, color=colors)
ax.set_xticks(range(len(mental_counts)))
ax.set_xticklabels(['Signs of Mental Illness', 'No Signs'], fontsize=12)
ax.set_ylabel('Number of Incidents', fontsize=12)
ax.set_title('Mental Health Factor: Prevalence in Fatal Encounters', fontsize=16, fontweight='bold', pad=20)
for i, (idx, val) in enumerate(mental_counts.items()):
    percentage = val / len(deaths) * 100
    ax.text(i, val + 20, f'{val:,}\n({percentage:.1f}%)', ha='center', fontsize=12, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(output_dir / "06_mental_illness_indicator.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# CHART 7: Threat Level Assessment
# ============================================================================
print("Generating Chart 7: Threat level...")
fig, ax = plt.subplots(figsize=(12, 6))
threat_counts = deaths['threat_level'].value_counts()
colors = ['#d62728', '#ff7f0e', '#2ca02c']
bars = ax.bar(range(len(threat_counts)), threat_counts.values, color=colors)
ax.set_xticks(range(len(threat_counts)))
ax.set_xticklabels([label.title() for label in threat_counts.index], fontsize=12)
ax.set_ylabel('Number of Incidents', fontsize=12)
ax.set_title('Threat Assessment: Perceived Threat Level During Incidents', fontsize=16, fontweight='bold', pad=20)
for i, (idx, val) in enumerate(threat_counts.items()):
    percentage = val / len(deaths) * 100
    ax.text(i, val + 20, f'{val:,}\n({percentage:.1f}%)', ha='center', fontsize=11, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(output_dir / "07_threat_level.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# CHART 8: Fleeing Status
# ============================================================================
print("Generating Chart 8: Fleeing status...")
fig, ax = plt.subplots(figsize=(12, 6))
flee_counts = deaths['flee'].value_counts()
colors = sns.color_palette("Set2", len(flee_counts))
bars = ax.bar(range(len(flee_counts)), flee_counts.values, color=colors)
ax.set_xticks(range(len(flee_counts)))
ax.set_xticklabels(flee_counts.index, fontsize=11)
ax.set_ylabel('Number of Incidents', fontsize=12)
ax.set_title('Flight Response: Subject Behavior During Encounter', fontsize=16, fontweight='bold', pad=20)
for i, (idx, val) in enumerate(flee_counts.items()):
    percentage = val / len(deaths) * 100
    ax.text(i, val + 15, f'{val:,}\n({percentage:.1f}%)', ha='center', fontsize=10, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(output_dir / "08_fleeing_status.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# CHART 9: Body Camera Presence
# ============================================================================
print("Generating Chart 9: Body camera usage...")
fig, ax = plt.subplots(figsize=(10, 6))
camera_counts = deaths['body_camera'].value_counts()
colors = ['#1f77b4', '#ff7f0e']
bars = ax.bar(range(len(camera_counts)), camera_counts.values, color=colors)
ax.set_xticks(range(len(camera_counts)))
ax.set_xticklabels(['No Body Camera', 'Body Camera Present'], fontsize=12)
ax.set_ylabel('Number of Incidents', fontsize=12)
ax.set_title('Transparency Tool: Body Camera Deployment Rate', fontsize=16, fontweight='bold', pad=20)
for i, (idx, val) in enumerate(camera_counts.items()):
    percentage = val / len(deaths) * 100
    ax.text(i, val + 30, f'{val:,}\n({percentage:.1f}%)', ha='center', fontsize=12, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(output_dir / "09_body_camera_presence.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# CHART 10: Gender Distribution
# ============================================================================
print("Generating Chart 10: Gender distribution...")
fig, ax = plt.subplots(figsize=(10, 6))
gender_counts = deaths['gender'].value_counts()
gender_labels = {'M': 'Male', 'F': 'Female'}
gender_counts.index = gender_counts.index.map(lambda x: gender_labels.get(x, 'Unknown'))
colors = ['#1f77b4', '#e377c2']
bars = ax.bar(range(len(gender_counts)), gender_counts.values, color=colors)
ax.set_xticks(range(len(gender_counts)))
ax.set_xticklabels(gender_counts.index, fontsize=12)
ax.set_ylabel('Number of Incidents', fontsize=12)
ax.set_title('Gender Distribution: Incidents by Gender', fontsize=16, fontweight='bold', pad=20)
for i, (idx, val) in enumerate(gender_counts.items()):
    percentage = val / len(deaths) * 100
    ax.text(i, val + 30, f'{val:,}\n({percentage:.1f}%)', ha='center', fontsize=12, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(output_dir / "10_gender_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# CHART 11: Manner of Death
# ============================================================================
print("Generating Chart 11: Manner of death...")
fig, ax = plt.subplots(figsize=(12, 6))
manner_counts = deaths['manner_of_death'].value_counts()
colors = sns.color_palette("Reds", len(manner_counts))
bars = ax.bar(range(len(manner_counts)), manner_counts.values, color=colors)
ax.set_xticks(range(len(manner_counts)))
ax.set_xticklabels([label.title() for label in manner_counts.index], rotation=15, ha='right', fontsize=11)
ax.set_ylabel('Number of Incidents', fontsize=12)
ax.set_title('Method Analysis: Classification of Fatal Force Used', fontsize=16, fontweight='bold', pad=20)
for i, (idx, val) in enumerate(manner_counts.items()):
    percentage = val / len(deaths) * 100
    ax.text(i, val + 15, f'{val:,}\n({percentage:.1f}%)', ha='center', fontsize=10, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(output_dir / "11_manner_of_death.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# CHART 12: Race vs Mental Illness - Stacked Analysis
# ============================================================================
print("Generating Chart 12: Race and mental illness correlation...")
fig, ax = plt.subplots(figsize=(14, 7))
race_mental = pd.crosstab(deaths['race'], deaths['signs_of_mental_illness'])
race_mental.index = race_mental.index.map(lambda x: race_labels.get(x, 'Unknown'))
race_mental_pct = race_mental.div(race_mental.sum(axis=1), axis=0) * 100
race_mental_pct = race_mental_pct.sort_values(True, ascending=False)

x = np.arange(len(race_mental_pct))
width = 0.6
p1 = ax.bar(x, race_mental_pct[True], width, label='Mental Illness Signs', color='#d62728')
p2 = ax.bar(x, race_mental_pct[False], width, bottom=race_mental_pct[True], label='No Signs', color='#2ca02c')

ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('Mental Health Context: Prevalence by Demographic Group', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(race_mental_pct.index, rotation=45, ha='right')
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(output_dir / "12_race_mental_illness.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# CHART 13: Top Cities Analysis
# ============================================================================
print("Generating Chart 13: Top cities...")
fig, ax = plt.subplots(figsize=(14, 8))
city_counts = deaths['city'].value_counts().head(20)
colors = sns.color_palette("Blues_r", len(city_counts))
bars = ax.barh(range(len(city_counts)), city_counts.values, color=colors)
ax.set_yticks(range(len(city_counts)))
ax.set_yticklabels(city_counts.index)
ax.set_xlabel('Number of Fatal Incidents', fontsize=12)
ax.set_title('Urban Concentration: Cities with Highest Incident Frequency', fontsize=16, fontweight='bold', pad=20)
ax.invert_yaxis()
for i, (idx, val) in enumerate(city_counts.items()):
    ax.text(val + 0.5, i, f'{val}', va='center', fontsize=10, fontweight='bold')
ax.grid(True, axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(output_dir / "13_top_cities.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# CHART 14: Armed vs Unarmed by Race
# ============================================================================
print("Generating Chart 14: Armed status by race...")
fig, ax = plt.subplots(figsize=(14, 7))
armed_binary = deaths.copy()
armed_binary['armed_status'] = armed_binary['armed'].apply(lambda x: 'Unarmed' if x == 'unarmed' else 'Armed')
race_armed = pd.crosstab(armed_binary['race'], armed_binary['armed_status'])
race_armed.index = race_armed.index.map(lambda x: race_labels.get(x, 'Unknown'))
race_armed = race_armed.loc[race_armed.sum(axis=1).sort_values(ascending=False).index]

x = np.arange(len(race_armed))
width = 0.35
p1 = ax.bar(x - width/2, race_armed['Armed'], width, label='Armed', color='#ff7f0e')
p2 = ax.bar(x + width/2, race_armed['Unarmed'], width, label='Unarmed', color='#1f77b4')

ax.set_ylabel('Number of Incidents', fontsize=12)
ax.set_title('Weapon Status: Armed vs Unarmed by Demographic Group', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(race_armed.index, rotation=45, ha='right')
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(output_dir / "14_armed_status_by_race.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# CHART 15: Quarterly Trend Analysis
# ============================================================================
print("Generating Chart 15: Quarterly trends...")
fig, ax = plt.subplots(figsize=(14, 6))
quarterly_counts = deaths.groupby(deaths['date'].dt.to_period('Q')).size()
quarterly_counts.index = quarterly_counts.index.to_timestamp()
ax.bar(range(len(quarterly_counts)), quarterly_counts.values, color='#9467bd', alpha=0.7, edgecolor='black')
ax.set_xticks(range(0, len(quarterly_counts), 2))
ax.set_xticklabels([f"Q{q.quarter} {q.year}" for q in quarterly_counts.index[::2]], rotation=45, ha='right')
ax.set_ylabel('Number of Incidents', fontsize=12)
ax.set_title('Quarterly Performance: Incident Volume Trends', fontsize=16, fontweight='bold', pad=20)
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(output_dir / "15_quarterly_trends.png", dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "="*70)
print("Chart generation completed successfully!")
print(f"Total charts created: 15")
print(f"Output directory: {output_dir.absolute()}")
print("="*70)
