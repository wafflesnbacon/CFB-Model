import streamlit as st
import cfbd
import pandas as pd
from sklearn.linear_model import LogisticRegression

# --- UI SETUP ---
st.set_page_config(page_title="CFB Win Predictor", layout="wide")
st.title("🏈 College Football Win Predictor")

# Uses Streamlit Cloud Secrets if deployed; falls back to local key on your PC
try:
    API_KEY = st.secrets["CFBD_API_KEY"]
except Exception:
    API_KEY = "YOUR_API_KEY_HERE"

YEARS = [2026]

# --- CACHED DATA ENGINE ---
@st.cache_data
def load_and_train_model():
    configuration = cfbd.Configuration(access_token=API_KEY)
    with cfbd.ApiClient(configuration) as api_client:
        betting_api = cfbd.BettingApi(api_client)
        stats_api = cfbd.StatsApi(api_client)
        games_api = cfbd.GamesApi(api_client)
        teams_api = cfbd.TeamsApi(api_client)
        
        # 1. Fetch Teams (Logos and Hex Colors)
        teams_data = teams_api.get_teams()
        team_dict = {}
        for t in teams_data:
            if t.classification == 'fbs':
                logo = t.logos[0] if t.logos else None
                color = t.color if t.color else "#000000"
                team_dict[t.school] = {'logo': logo, 'color': color}
                
        lines_data = []
        scores_data = []
        stats_data = []
        luck_records = []

        # 2. Fetch Data Across Years
        for yr in YEARS:
            lines = betting_api.get_lines(year=yr, season_type='regular')
            for g in lines:
                if g.lines:
                    line_obj = next((x for x in g.lines if x.provider == 'consensus'), g.lines[0])
                    if line_obj.spread is not None:
                        lines_data.append({
                            'game_id': g.id, 
                            'year': yr,
                            'home_team': g.home_team, 
                            'away_team': g.away_team, 
                            'spread_close': float(line_obj.spread)
                        })

            games = games_api.get_games(year=yr, season_type='regular', classification='fbs')
            for g in games:
                if g.home_points is not None and g.away_points is not None:
                    scores_data.append({
                        'game_id': g.id, 
                        'home_score': g.home_points, 
                        'away_score': g.away_points, 
                        'is_home_game': 0 if g.neutral_site else 1
                    })
                    
                    # Extract Postgame Win Probabilities for Luck Calculation
                    home_pgwp = getattr(g, 'home_postgame_win_probability', None)
                    away_pgwp = getattr(g, 'away_postgame_win_probability', None)
                    if home_pgwp is not None and away_pgwp is not None:
                        home_won = 1 if g.home_points > g.away_points else 0
                        luck_records.append({
                            'year': yr, 'week': g.week, 'team': g.home_team,
                            'actual_win': home_won, 'expected_win': float(home_pgwp)
                        })
                        luck_records.append({
                            'year': yr, 'week': g.week, 'team': g.away_team,
                            'actual_win': 1 - home_won, 'expected_win': float(away_pgwp)
                        })

            stats = stats_api.get_advanced_game_stats(year=yr, exclude_garbage_time=True)
            for s in stats:
                stats_data.append({
                    'game_id': s.game_id, 
                    'year': yr,
                    'team': s.team, 
                    'week': s.week, 
                    'off_epa': s.offense.ppa if s.offense else 0, 
                    'def_epa': s.defense.ppa if s.defense else 0,
                    'off_std_sr': s.offense.standard_downs.success_rate if (s.offense and s.offense.standard_downs) else 0
                })

        df_lines = pd.DataFrame(lines_data)
        df_scores = pd.DataFrame(scores_data)
        df_stats = pd.DataFrame(stats_data)
        df_luck = pd.DataFrame(luck_records)

    # 3. Calculate 6-Game Rolling Averages (Chronological across seasons)
    span = 6
    df_stats = df_stats.sort_values(by=['team', 'year', 'week'])
    df_stats['rolling_off_epa'] = df_stats.groupby('team')['off_epa'].transform(lambda x: x.ewm(span=span).mean().shift(1))
    df_stats['rolling_def_epa'] = df_stats.groupby('team')['def_epa'].transform(lambda x: x.ewm(span=span).mean().shift(1))
    df_stats['rolling_off_std_sr'] = df_stats.groupby('team')['off_std_sr'].transform(lambda x: x.ewm(span=span).mean().shift(1))
    df_stats = df_stats.dropna(subset=['rolling_off_epa'])

    # 4. Structure Matrices & Train Champion Model
    df_home = df_stats.rename(columns={'team': 'home_team', 'rolling_off_epa': 'home_off_epa', 'rolling_def_epa': 'home_def_epa', 'rolling_off_std_sr': 'home_std_sr'})
    df_away = df_stats.rename(columns={'team': 'away_team', 'rolling_off_epa': 'away_off_epa', 'rolling_def_epa': 'away_def_epa', 'rolling_off_std_sr': 'away_std_sr'})
    df = df_lines.merge(df_scores, on='game_id').merge(df_home, on=['game_id', 'home_team']).merge(df_away, on=['game_id', 'away_team'])

    df['epa_diff'] = (df['home_off_epa'] - df['away_def_epa']) - (df['away_off_epa'] - df['home_def_epa'])
    df['std_sr_diff'] = df['home_std_sr'] - df['away_std_sr']
    df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
    df = df.dropna()

    ml_features = ['spread_close', 'epa_diff', 'std_sr_diff', 'is_home_game']
    model = LogisticRegression()
    model.fit(df[ml_features], df['home_win'])
    
    # Filter stats & luck down to the active season (2026) for the UI slider
    active_year = 2026 if not df_stats[df_stats['year'] == 2026].empty else 2025
    df_current_season = df_stats[df_stats['year'] == active_year]
    df_current_luck = df_luck[df_luck['year'] == active_year] if not df_luck.empty else pd.DataFrame()
        
    return model, df_current_season, df_current_luck, team_dict

# --- LOAD DATA ---
with st.spinner('Crunching matrices and pulling APIs...'):
    model, df_stats, df_luck, team_dict = load_and_train_model()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Matchup Settings")
team_list = sorted(list(team_dict.keys()))
default_home = team_list.index("Texas") if "Texas" in team_list else 0

# Time Machine Slider
max_week = int(df_stats['week'].max())
selected_week = st.sidebar.slider(
    "Simulate Week (Time Machine)", 
    min_value=1, 
    max_value=max_week, 
    value=max_week, 
    help="Select the week to see the prediction using only data available up to that week."
)

# Full-Width Team Dropdowns
away_team = st.sidebar.selectbox("Away Team", team_list, index=0)
home_team = st.sidebar.selectbox("Home Team", team_list, index=default_home)

# Side-by-Side Live Logo Preview in Sidebar
logo_away = team_dict[away_team]['logo'] or ""
logo_home = team_dict[home_team]['logo'] or ""
st.sidebar.markdown(f"""
<div style="display: flex; justify-content: space-around; align-items: center; background-color: #1E1E1E; padding: 10px; border-radius: 8px; margin-bottom: 15px;">
    <div style="text-align: center;">
        <img src="{logo_away}" width="45" onerror="this.style.display='none'"><br>
        <span style="font-size: 12px; color: #CCCCCC;">Away</span>
    </div>
    <span style="font-weight: bold; color: #888888;">@</span>
    <div style="text-align: center;">
        <img src="{logo_home}" width="45" onerror="this.style.display='none'"><br>
        <span style="font-size: 12px; color: #CCCCCC;">Home</span>
    </div>
</div>
""", unsafe_allow_html=True)

vegas_spread = st.sidebar.number_input("Vegas Spread (Negative = Home Favored)", value=-3.5, step=0.5)
is_neutral = st.sidebar.checkbox("Neutral Site Game", value=False)

# --- HELPER FUNCTION FOR LUCKINESS ---
def get_luck_badge(team_name, week_limit):
    if df_luck.empty:
        return "0.00", "Neutral", "#888888"
    past_games = df_luck[(df_luck['team'] == team_name) & (df_luck['week'] < week_limit)]
    if past_games.empty:
        return "0.00", "No Prior Games", "#888888"
    
    luck_val = past_games['actual_win'].sum() - past_games['expected_win'].sum()
    if luck_val >= 0.5:
        return f"+{luck_val:.2f} Wins", "Lucky 🍀", "#4CAF50"
    elif luck_val <= -0.5:
        return f"{luck_val:.2f} Wins", "Unlucky 💀", "#FF5252"
    else:
        sign = "+" if luck_val > 0 else ""
        return f"{sign}{luck_val:.2f} Wins", "As Expected", "#B0BEC5"

# --- PREDICTION ENGINE ---
if st.sidebar.button("Predict Matchup"):
    home_past = df_stats[(df_stats['team'] == home_team) & (df_stats['week'] <= selected_week)]
    away_past = df_stats[(df_stats['team'] == away_team) & (df_stats['week'] <= selected_week)]
    
    if home_past.empty or away_past.empty:
        st.error(f"Error: Not enough historical data up to Week {selected_week} to generate a prediction.")
    else:
        home_stats = home_past.iloc[-1]
        away_stats = away_past.iloc[-1]
        
        home_adj_epa = home_stats['rolling_off_epa'] - away_stats['rolling_def_epa']
        away_adj_epa = away_stats['rolling_off_epa'] - home_stats['rolling_def_epa']
        
        epa_diff = home_adj_epa - away_adj_epa
        std_sr_diff = home_stats['rolling_off_std_sr'] - away_stats['rolling_off_std_sr']
        is_home = 0 if is_neutral else 1

        matchup_data = pd.DataFrame({
            'spread_close': [vegas_spread], 
            'epa_diff': [epa_diff], 
            'std_sr_diff': [std_sr_diff], 
            'is_home_game': [is_home]
        })
        
        win_prob = model.predict_proba(matchup_data)[0]
        away_prob = win_prob[0] * 100
        home_prob = win_prob[1] * 100

        # Calculate Luck Factors up to selected week
        away_luck_str, away_luck_label, away_luck_color = get_luck_badge(away_team, selected_week)
        home_luck_str, home_luck_label, home_luck_color = get_luck_badge(home_team, selected_week)

        # --- VISUAL DASHBOARD ---
        st.markdown("---")
        c1 = team_dict[away_team]['color']
        c2 = team_dict[home_team]['color']
        
        col1, col2, col3 = st.columns([1.2, 0.6, 1.2])

        # Away Team Header + Luck Badge
        with col1:
            logo_html_1 = f"<img src='{team_dict[away_team]['logo']}' width='90' onerror=\"this.style.display='none'\">" if team_dict[away_team]['logo'] else ""
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 15px;">
                {logo_html_1}
                <div>
                    <h3 style="margin: 0; color: {c1};">{away_team}</h3>
                    <div style="background-color: #1E1E1E; border: 1px solid {away_luck_color}; border-radius: 6px; padding: 4px 10px; margin-top: 6px; display: inline-block;">
                        <span style="color: #AAAAAA; font-size: 12px; text-transform: uppercase;">Luck Factor: </span>
                        <span style="color: {away_luck_color}; font-weight: bold; font-size: 14px;">{away_luck_str} ({away_luck_label})</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Home Team Header + Luck Badge (Right-Aligned)
        with col3:
            logo_html_2 = f"<img src='{team_dict[home_team]['logo']}' width='90' onerror=\"this.style.display='none'\">" if team_dict[home_team]['logo'] else ""
            st.markdown(f"""
            <div style="display: flex; align-items: center; justify-content: flex-end; gap: 15px; text-align: right;">
                <div>
                    <h3 style="margin: 0; color: {c2};">{home_team}</h3>
                    <div style="background-color: #1E1E1E; border: 1px solid {home_luck_color}; border-radius: 6px; padding: 4px 10px; margin-top: 6px; display: inline-block;">
                        <span style="color: #AAAAAA; font-size: 12px; text-transform: uppercase;">Luck Factor: </span>
                        <span style="color: {home_luck_color}; font-weight: bold; font-size: 14px;">{home_luck_str} ({home_luck_label})</span>
                    </div>
                </div>
                {logo_html_2}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        st.markdown(f"#### Win Probability (Entering Week {selected_week})")
        progress_html = f"""
        <div style="width: 100%; display: flex; height: 45px; border-radius: 8px; overflow: hidden; box-shadow: 0px 4px 6px rgba(0,0,0,0.3); margin-bottom: 30px;">
            <div style="width: {away_prob}%; background-color: {c1}; display: flex; align-items: center; justify-content: flex-start; padding-left: 15px; color: white; font-weight: bold; font-size: 20px; text-shadow: 1px 1px 2px black;">
                {away_prob:.1f}%
            </div>
            <div style="width: {home_prob}%; background-color: {c2}; display: flex; align-items: center; justify-content: flex-end; padding-right: 15px; color: white; font-weight: bold; font-size: 20px; text-shadow: 1px 1px 2px black;">
                {home_prob:.1f}%
            </div>
        </div>
        """
        st.markdown(progress_html, unsafe_allow_html=True)
        
        st.markdown("#### Advanced Metric Head-to-Head")
        def create_metric_card(label, val1, val2, higher_is_better=True):
            if (val1 > val2 and higher_is_better) or (val1 < val2 and not higher_is_better):
                w1, w2, col1, col2 = "bold", "normal", c1, "gray"
            else:
                w1, w2, col1, col2 = "normal", "bold", "gray", c2
                
            return f"""
            <div style="background-color: #1E1E1E; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 6px solid {col1}; border-right: 6px solid {col2}; display: flex; justify-content: space-between; align-items: center; box-shadow: 0px 2px 4px rgba(0,0,0,0.2);">
                <div style="width: 25%; text-align: left; color: {col1}; font-weight: {w1}; font-size: 22px;">{val1:.3f}</div>
                <div style="width: 50%; text-align: center; color: #E0E0E0; font-size: 16px; font-weight: bold; text-transform: uppercase; letter-spacing: 1px;">{label}</div>
                <div style="width: 25%; text-align: right; color: {col2}; font-weight: {w2}; font-size: 22px;">{val2:.3f}</div>
            </div>
            """

        st.markdown(create_metric_card("Adj. Off EPA / Play", away_adj_epa, home_adj_epa), unsafe_allow_html=True)
        st.markdown(create_metric_card("Standard Down Success Rate", away_stats['rolling_off_std_sr'], home_stats['rolling_off_std_sr']), unsafe_allow_html=True)