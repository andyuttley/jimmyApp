import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import os


st.set_page_config(layout="wide")

# Inject custom CSS to reduce padding
st.markdown("""
    <style>
    .main {
        padding-left: 0rem;
        padding-right: 0rem;
    }
    </style>
    """, unsafe_allow_html=True)


st.header("Jimmy Bullard - ManeBall Analysis")

# Load data
df = pd.read_csv('final_df.csv')
granular_results = pd.read_csv('granular_results.csv')
finishing_positions = pd.read_csv('finishing_positions.csv')
top_4_odds = pd.read_csv('top_4_odds.csv')
score_distributions = pd.read_pickle('score_distributions.pkl')


max_gameweek = df['Gameweek'].max()  # Find the maximum Gameweek number

###############
# CURRENT TABLE
###############
with st.expander(":trophy: Current Table :trophy:", expanded=False):
    st.subheader(f"Current Table after {max_gameweek} Gameweeks")

    aggregated_table = df.groupby('Player').agg({
        'Table Points': 'sum',
        'Luck Points': 'sum',
        'Player_Score': 'sum',
        'Opponent_Score': 'sum'
    }).reset_index()

    # Rank by Table Points (descending) and Player Score (secondary)
    aggregated_table = aggregated_table.sort_values(
        by=['Table Points', 'Player_Score'], ascending=[False, False]
    )

    st.write("Aggregated Stats:")
    st.dataframe(aggregated_table)

##############################
# PREVIEW GAMEWEEK
##############################

# Load fixtures file dynamically
fixtures_file = 'fixtures.csv'
if os.path.exists(fixtures_file):
    fixtures = pd.read_csv(fixtures_file)
else:
    st.error(f"File '{fixtures_file}' not found. Please upload it.")
    st.stop()

# Get the minimum gameweek
min_gameweek = fixtures['Gameweek'].min()

# Filter out duplicate fixtures (e.g., Tim vs Mikey and Mikey vs Tim)
fixtures['Pair'] = fixtures.apply(lambda x: tuple(sorted([x['Player'], x['Opponent']])), axis=1)
fixtures = fixtures.drop_duplicates(subset=['Pair'])

# PREVIEW GAMEWEEK X
with st.expander(f":calendar: Preview Gameweek {min_gameweek} :calendar:", expanded=False):
    st.subheader(f"Fixtures for Gameweek {min_gameweek}")

    # Summary section
    st.markdown("### Summary")
    summary_lines = []

    # Function to calculate form-based probabilities
    def calculate_form_probabilities(player, opponent, df, max_gameweek):
        # Consider the last 5 gameweeks, with higher weights for more recent weeks
        recent_gameweeks = range(max_gameweek - 4, max_gameweek + 1)

        player_form = df[(df['Player'] == player) & (df['Gameweek'].isin(recent_gameweeks))]
        opponent_form = df[(df['Player'] == opponent) & (df['Gameweek'].isin(recent_gameweeks))]

        if player_form.empty or opponent_form.empty:
            return None, None, None

        # Assign weights to recent gameweeks
        weights = np.linspace(1, 5, num=5)  # Linearly increasing weights

        # Calculate weighted average scores
        player_weighted_score = np.average(player_form['Player_Score'].values, weights=weights)
        opponent_weighted_score = np.average(opponent_form['Player_Score'].values, weights=weights)

        # Simulate probabilities based on weighted scores
        total = player_weighted_score + opponent_weighted_score
        win_prob = round((player_weighted_score / total) * 100, 1)
        lose_prob = round((opponent_weighted_score / total) * 100, 1)
        draw_prob = round(100 - win_prob - lose_prob, 1)  # Remaining percentage for a draw

        return win_prob, draw_prob, lose_prob

    # Filter for the minimum gameweek
    gw_fixtures = fixtures[fixtures['Gameweek'] == min_gameweek]

    for _, row in gw_fixtures.iterrows():
        player = row['Player']
        opponent = row['Opponent']

        # Form-based prediction
        win_prob, draw_prob, lose_prob = calculate_form_probabilities(player, opponent, df, max_gameweek)
        if win_prob is not None:
            predicted_winner = player if win_prob > lose_prob else opponent
            summary_lines.append(f"- {player} vs {opponent}: Predicted winner is {predicted_winner}")
        else:
            summary_lines.append(f"- {player} vs {opponent}: Insufficient data for prediction")

    st.markdown("\n".join(summary_lines))

    # Display each fixture with details
    for _, row in gw_fixtures.iterrows():
        player = row['Player']
        opponent = row['Opponent']

        st.subheader(f"{player} vs {opponent}")

        # Historical head-to-head
        history = df[(df['Player'] == player) & (df['Opponent'] == opponent)]
        total_matches = len(history)
        player_wins = (history['Result'] == 'Win').sum()
        opponent_wins = (history['Result'] == 'Lose').sum()
        draws = (history['Result'] == 'Draw').sum()

        st.write(f"They have played {total_matches} times before: {player} has won {player_wins}, {opponent} has won {opponent_wins}, and they drew {draws} times.")

        # Weekly hypothetical head-to-head
        player_scores = df[df['Player'] == player]['Player_Score']
        opponent_scores = df[df['Player'] == opponent]['Player_Score']

        if not player_scores.empty and not opponent_scores.empty:
            total_weeks = min(len(player_scores), len(opponent_scores))
            # Align indices to ensure comparisons work
            player_scores = player_scores.reset_index(drop=True)
            opponent_scores = opponent_scores.reset_index(drop=True)
            
            # Trim both series to the same length
            length = min(len(player_scores), len(opponent_scores))
            player_scores = player_scores[:length]
            opponent_scores = opponent_scores[:length]
            
            # Compute weekly wins and draws
            player_weekly_wins = (player_scores > opponent_scores).sum()
            opponent_weekly_wins = (opponent_scores > player_scores).sum()
            weekly_draws = (player_scores == opponent_scores).sum()


            st.write(f"If they played every week, {player} would have won {player_weekly_wins / total_weeks:.1%} of the matches, {opponent} would have won {opponent_weekly_wins / total_weeks:.1%}, and {weekly_draws / total_weeks:.1%} would have been draws.")

            # Line chart of scores
            combined_scores = pd.DataFrame({
                'Gameweek': range(1, total_weeks + 1),
                player: player_scores[:total_weeks].values,
                opponent: opponent_scores[:total_weeks].values
            })

            line_chart = alt.Chart(combined_scores).mark_line().encode(
                x=alt.X('Gameweek:Q', title='Gameweek'),
                y=alt.Y('value:Q', title='Score'),
                color=alt.Color('variable:N', title='Player')
            ).transform_fold(
                [player, opponent],
                as_=['variable', 'value']
            ).properties(
                width=800, height=400
            )

            st.altair_chart(line_chart, use_container_width=True)

        # Form-based prediction
        win_prob, draw_prob, lose_prob = calculate_form_probabilities(player, opponent, df, max_gameweek)
        if win_prob is not None:
            st.write(f"Form suggests: In the previous 5 gameweeks, giving increasing weight to more recent gameweeks, it would suggest that {player} has a {win_prob}% chance of winning, {draw_prob}% chance of drawing, and {lose_prob}% chance of losing.")
        else:
            st.write("Form-based prediction unavailable due to insufficient data.")



##############################
# 1000 SIMULATED SEASON ENDS
##############################
with st.expander(":robot_face: Simulating 1000 seasons... :robot_face:", expanded=False):
    st.subheader("1000 Simulated Season Ends")
    st.write("*1000 simulated season endings are generated by modeling the outcome of all remaining fixtures, over and over again. This approach incorporates both individual player histories and league-wide baselines, applying a form of regression to the mean that tempers early-season extremes.*")
    st.write("")
    st.write("*A truncated normal distribution, slightly extended beyond historical bounds, ensures that while improbable scores remain rare, they are not impossible, maintaining a realistic spread of outcomes. Each player's mean and standard deviation blend their own data with the league's, so that a handful of lucky or unlucky weeks do not disproportionately dictate future potential. By repeatedly drawing from these distributions, the simulations produce a probability landscape for final standings that recognizes both individual variance and the shared statistical tendencies of the entire league. As a result, the model yields - in theory - a broad set of the possible season conclusions. This shows us the all those possible outcomes, and how likely they actually are across 1000 seasons.*")

    st.write("Statistically, the most likely Playoff matches right now are:")

    # Sort top_4_odds by 'Top 4 %' descending
    top_4_odds_sorted = top_4_odds.sort_values(by='Top 4 %', ascending=False)
    top4_players = top_4_odds_sorted.head(4)['Player'].tolist()

    if len(top4_players) == 4:
        st.write(f"- {top4_players[0]} vs {top4_players[3]}")
        st.write(f"- {top4_players[1]} vs {top4_players[2]}")
    else:
        st.write("Not enough players to form top 4 playoff matches.")


    # Show the full top_4_odds table
    st.write("Full Top 4 Odds:")
    st.dataframe(top_4_odds_sorted)

    # Show a chart over time of each player's changing % likelihood to make playoffs
    history_filename = "top_4_odds_history.csv"
    if os.path.exists(history_filename):
        top_4_history = pd.read_csv(history_filename)
        st.write("Top 4 Odds Over Time:")
        line_chart = alt.Chart(top_4_history).mark_line(point=True).encode(
            x=alt.X('as of finished_gw:O', title='Finished Gameweek'),
            y=alt.Y('Top 4 %:Q', title='Top 4 Probability (%)'),
            color='Player:N',
            tooltip=['Player', 'Top 4 %', 'as of finished_gw']
        ).properties(
            width=800,
            height=400,
            title="Changes in Top 4 Probability Over Time"
        )
        st.altair_chart(line_chart, use_container_width=True)
    else:
        st.write("No historical top_4_odds data available yet.")

    # Show finishing_positions table
    st.write("Finishing Positions (Percent Chance at Each Rank):")
    st.dataframe(finishing_positions)
    
    
    # Create a single distribution chart (lines) of each player's 'bag' of scores
    scores_list = []
    for player, arr in score_distributions.items():
        for val in arr:
            scores_list.append((player, val))
    scores_df = pd.DataFrame(scores_list, columns=['Player', 'Score'])

    st.write("Distribution of Scores (All Players):")
    # Use Altair's transform_density with a fixed x domain [10,90]
    density_chart = alt.Chart(scores_df).transform_density(
        density='Score',
        groupby=['Player'],
        as_=['Score', 'density']
    ).mark_line().encode(
        x=alt.X('Score:Q', scale=alt.Scale(domain=[10,90]), title='Score'),
        y=alt.Y('density:Q', title='Density'),
        color='Player:N',
        tooltip=['Player', alt.Tooltip('Score:Q'), alt.Tooltip('density:Q')]
    ).properties(
        width=800,
        height=400,
        title="Score Distribution Lines for Each Player"
    )
    st.altair_chart(density_chart, use_container_width=True)

    # Single chart for an individual player's distribution, chosen by a selectbox
    st.write("## See your 10,000 bag of scores fit to your historic distribution")
    default_player = "Andy" if "Andy" in scores_df['Player'].unique() else scores_df['Player'].unique()[0]
    selected_player = st.selectbox("Select a Player", sorted(scores_df['Player'].unique()), index=sorted(scores_df['Player'].unique()).index(default_player))

    player_subset = scores_df[scores_df['Player'] == selected_player]

    hist = alt.Chart(player_subset).mark_bar(opacity=0.5, stroke='black').encode(
        x=alt.X('Score:Q', bin=alt.Bin(step=1), scale=alt.Scale(domain=[0,100]), title='Score'),
        y=alt.Y('count()', title='Frequency')
    )

    density_line = alt.Chart(player_subset).transform_density(
        density='Score',
        as_=['Score', 'density']
    ).mark_line(color='red').encode(
        x=alt.X('Score:Q', scale=alt.Scale(domain=[0,100]), title='Score'),
        y=alt.Y('density:Q', axis=alt.Axis(title='Density')),
        tooltip=[alt.Tooltip('Score:Q'), alt.Tooltip('density:Q')]
    )

    combined = alt.layer(hist, density_line).resolve_scale(y='independent').properties(
        width=800,
        height=250,
        title=f"{selected_player}'s Score Distribution"
    )

    st.altair_chart(combined, use_container_width=True)

with st.expander("🍀 What's LUCK got to do with it? 🍀", expanded=False):
    st.write("""
    *Luck points are derived by comparing each player's weekly performance to the entire league WITHIN that same gameweek. For example, you could be the second highest scorer of the week, but if you were playing the highest scorer of that week, you'd be deemed unlucky.
    By taking a strict rules-based approach, a season-long aggregate luck score emerges. These rules are defined as: top 4 scoring players within a week should win, the 5th and 6th should draw, the other should lose.
    This is, of course, ONE way of 'defining luck' - you may have your own ideas. This is just one proxy for it. Of course, we do not get on average one draw a week, but this at least introduces some decay Vs the 1st and 5th highest scorers each week both just blanket winning, and is a possible sweet spot.*
    \n *This process highlights players whose actual results deviate significantly from their expected outcomes, given their weekly rank and the distribution of scores around them. 
    The final Luck Adjusted Pos is simply the rank order of players by Luck Points instead of Table Points, and we can then compare this rank to their actual position, revealing who has benefitted from fortune and who has been hindered by misfortune.*""")

    # Aggregate data from final_df
    agg = df.groupby('Player').agg({
        'Table Points':'sum',
        'Luck Points':'sum',
        'Player_Score':'sum'
    }).reset_index()

    # Determine actual positions (by Table Points descending)
    agg = agg.sort_values(['Table Points','Player_Score'], ascending=[False,False])
    agg['Actual Pos'] = range(1, len(agg)+1)

    # Determine luck-adjusted positions (by Luck Points descending)
    agg = agg.sort_values(['Luck Points','Player_Score'], ascending=[False,False])
    agg['Luck Adjusted Pos'] = range(1, len(agg)+1)

    # Compute Difference (Actual - Luck Adjusted)
    agg['Difference'] = agg['Actual Pos'] - agg['Luck Adjusted Pos']

    # Show the resulting table sorted by Luck Points descending
    agg = agg.sort_values(['Luck Points','Player_Score'], ascending=[False,False])

    st.dataframe(agg[['Player','Luck Points','Table Points','Player_Score','Actual Pos','Luck Adjusted Pos','Difference']])

    # Player GW Rank Matrix Section
    
    st.subheader("Player GW Rank Matrix")
    st.write("Matrix of the count each time each player has finished in that rank for the gameweek")

    # Create the matrix: rows are players, columns are GW ranks, values are counts
    rank_matrix = df.pivot_table(
        index='Player', 
        columns='GW Rank', 
        aggfunc='size', 
        fill_value=0
    )

    # Rename columns for clarity (e.g., 1 -> 1st, 2 -> 2nd, etc.)
    rank_matrix.columns = [f"{int(col)}{('th' if 4 <= col <= 20 else {1:'st',2:'nd',3:'rd'}.get(col % 10, 'th'))}" for col in rank_matrix.columns]

    # Show the matrix
    st.dataframe(rank_matrix, use_container_width=True)


###############
# SEASON AWARDS
###############
with st.expander(":sports_medal: Season Awards :sports_medal:", expanded=False):
    st.subheader("Season Awards")

    luck_stats = aggregated_table.copy()
    luck_stats['Net Luck'] = luck_stats['Luck Points'] - luck_stats['Table Points']

    luckiest = luck_stats.sort_values(by='Net Luck').head(1)
    unluckiest = luck_stats.sort_values(by='Net Luck', ascending=False).head(1)

    highest_scoring_week = df.loc[df['Player_Score'].idxmax()]
    lowest_scoring_week = df.loc[df['Player_Score'].idxmin()]

    player_stats = df.groupby('Player')['Player_Score'].agg(['std', 'mean']).reset_index()

    most_consistent = player_stats.sort_values(by='std').head(1)
    least_consistent = player_stats.sort_values(by='std', ascending=False).head(1)

    unlucky_losses = df[(df['Result'] == 'Lose')].sort_values(by='Player_Score', ascending=False)
    if not unlucky_losses.empty:
        most_unlucky_loss = unlucky_losses.iloc[0]

    highest_scorer_counts = df.groupby('Player')['GW Rank'].apply(lambda x: (x == 1).sum())
    most_highest_scorers = highest_scorer_counts[highest_scorer_counts == highest_scorer_counts.max()]

    lowest_scorer_counts = df.groupby('Player')['GW Rank'].apply(lambda x: (x == df['GW Rank'].max()).sum())
    most_lowest_scorers = lowest_scorer_counts[lowest_scorer_counts == lowest_scorer_counts.max()]

    avg_opponent_scores = df.groupby('Player')['Opponent_Score'].mean().reset_index()
    lowest_avg_opponent = avg_opponent_scores.sort_values(by='Opponent_Score').head(1)
    highest_avg_opponent = avg_opponent_scores.sort_values(by='Opponent_Score', ascending=False).head(1)

    luckiest_player = luckiest.iloc[0]['Player']
    luckiest_points = luckiest.iloc[0]['Net Luck']

    unluckiest_player = unluckiest.iloc[0]['Player']
    unluckiest_points = unluckiest.iloc[0]['Net Luck']

    highest_scorer = highest_scoring_week['Player']
    highest_score = highest_scoring_week['Player_Score']
    highest_week = highest_scoring_week['Gameweek']

    lowest_scorer = lowest_scoring_week['Player']
    lowest_score = lowest_scoring_week['Player_Score']
    lowest_week = lowest_scoring_week['Gameweek']

    most_consistent_player = most_consistent.iloc[0]['Player']
    most_consistent_std = most_consistent.iloc[0]['std']
    most_consistent_mean = most_consistent.iloc[0]['mean']

    least_consistent_player = least_consistent.iloc[0]['Player']
    least_consistent_std = least_consistent.iloc[0]['std']
    least_consistent_mean = least_consistent.iloc[0]['mean']

    if not unlucky_losses.empty:
        unlucky_loss_player = most_unlucky_loss['Player']
        unlucky_loss_score = most_unlucky_loss['Player_Score']
        unlucky_loss_opponent = most_unlucky_loss['Opponent']
        unlucky_loss_opponent_score = most_unlucky_loss['Opponent_Score']
        unlucky_loss_week = most_unlucky_loss['Gameweek']
    else:
        unlucky_loss_player = unlucky_loss_score = unlucky_loss_opponent = unlucky_loss_opponent_score = unlucky_loss_week = None

    highest_scorer_count = most_highest_scorers.max()
    highest_scorer_list = ', '.join(most_highest_scorers.index)

    lowest_scorer_count = most_lowest_scorers.max()
    lowest_scorer_list = ', '.join(most_lowest_scorers.index)

    lowest_avg_player = lowest_avg_opponent.iloc[0]['Player']
    lowest_avg_score = lowest_avg_opponent.iloc[0]['Opponent_Score']
    highest_avg_player = highest_avg_opponent.iloc[0]['Player']
    highest_avg_score = highest_avg_opponent.iloc[0]['Opponent_Score']
    advantage_gap = highest_avg_score - lowest_avg_score

    def luck_line(player_name, points, luckiest=True):
        val = int(round(abs(points)))
        if points < 0:
            if luckiest:
                return f":trophy: The **luckiest player** this season is **{player_name}**, who earned {val} points more than they deserved."
            else:
                return f":trophy: The **unluckiest player** is **{player_name}**, who earned {val} points more than they deserved."
        else:
            if luckiest:
                return f":trophy: The **luckiest player** this season is **{player_name}**, who earned {val} points fewer than they deserved."
            else:
                return f":trophy: The **unluckiest player** is **{player_name}**, who earned {val} points fewer than they deserved."

    summary_lines = []
    summary_lines.append(luck_line(luckiest_player, luckiest_points, luckiest=True))
    summary_lines.append(luck_line(unluckiest_player, unluckiest_points, luckiest=False))
    summary_lines.append(f":trophy: The **highest scoring week** belongs to **{highest_scorer}**, who scored {highest_score} points in Gameweek {highest_week}.")
    summary_lines.append(f":trophy: The **lowest scoring week** was by **{lowest_scorer}**, who scored just {lowest_score} points in Gameweek {lowest_week}.")
    summary_lines.append(f":trophy: The **most consistent player** is **{most_consistent_player}**, with a standard deviation of {most_consistent_std:.2f} and an average score of {most_consistent_mean:.2f}.")
    summary_lines.append(f":trophy: The **least consistent player** is **{least_consistent_player}**, with a standard deviation of {least_consistent_std:.2f} and an average score of {least_consistent_mean:.2f}.")

    if unlucky_loss_player:
        summary_lines.append(f":trophy: The **most unlucky loss** occurred in Gameweek {unlucky_loss_week}, where **{unlucky_loss_player}** scored {unlucky_loss_score} points but lost to **{unlucky_loss_opponent}**, who scored {unlucky_loss_opponent_score}.")

    summary_lines.append(f":trophy: The **most times as highest scorer** belongs to: **{highest_scorer_list}** ({highest_scorer_count} times).")
    summary_lines.append(f":trophy: The **most times as lowest scorer** belongs to: **{lowest_scorer_list}** ({lowest_scorer_count} times).")
    summary_lines.append(f":trophy: The **home advantage analysis** shows that **{lowest_avg_player}** has the lowest average opponent score ({lowest_avg_score:.2f}), while **{highest_avg_player}** faces the highest ({highest_avg_score:.2f}). **{lowest_avg_player}** would need to score {advantage_gap:.2f} fewer points on average each week to match this advantage.")

    for line in summary_lines:
        st.write(line)

#########################
# PLAYER VS PLAYER COMPARISON
#########################
with st.expander(":people_holding_hands: Player vs Player Comparison :people_holding_hands:", expanded=False):
    st.subheader("Player vs Player Comparison")
    st.write("See how you compare to another player, if you'd head-to-head played them each week.")

    players = df['Player'].unique()
    player1 = st.selectbox("Choose Player 1", players, index=0, key="player1_selectbox")
    player2 = st.selectbox("Choose Player 2", players, index=1, key="player2_selectbox")

    if player1 and player2:
        player1_scores = df[df['Player'] == player1][['Gameweek', 'Player_Score']]
        player2_scores = df[df['Player'] == player2][['Gameweek', 'Player_Score']]

        matchup = pd.merge(player1_scores, player2_scores, on='Gameweek', suffixes=('_P1', '_P2'))

        player1_wins = (matchup['Player_Score_P1'] > matchup['Player_Score_P2']).sum()
        player2_wins = (matchup['Player_Score_P2'] > matchup['Player_Score_P1']).sum()
        draws = (matchup['Player_Score_P1'] == matchup['Player_Score_P2']).sum()
        total_games = len(matchup)

        st.write(f"- **{player1}** would have won {player1_wins} of the matches ({player1_wins / total_games:.1%}).")
        st.write(f"- **{player2}** would have won {player2_wins} of the matches ({player2_wins / total_games:.1%}).")
        st.write(f"- They would have drawn {draws} of the matches ({draws / total_games:.1%}).")

##############################
# PLAYER SCORES BY GAMEWEEK
##############################
with st.expander(":chart_with_upwards_trend: Player Scoring Patterns :chart_with_upwards_trend:", expanded=False):
    st.subheader("Player Scores by Gameweek")

    # Compute avg losing score + 1
    avg_losing_score = df[df['Result'] == 'Lose']['Player_Score'].mean()
    line_value = avg_losing_score + 1

    chart_data = df[['Player', 'Gameweek', 'Player_Score', 'Result']].copy()
    chart_data['Gameweek'] = chart_data['Gameweek'].astype(int)  # Ensure Gameweek is numeric

    # Sort players based on the order of the aggregated table
    player_ranking = aggregated_table[['Player']].reset_index()
    player_ranking['Rank'] = player_ranking.index
    chart_data = chart_data.merge(player_ranking, on='Player', how='left')

    st.write("## Player scoring so far")
    st.write(f"Showing all scores so far, where the average points needed to win is approximately {line_value:.1f}")

    result_color_map = {
        'Win': 'green',
        'Lose': 'red',
        'Draw': 'grey'
    }
    chart_data['Result_Color'] = chart_data['Result'].map(result_color_map)

    # First chart: all scores per gameweek + rule for avg losing score +1
    scatter_chart = alt.Chart(chart_data).mark_circle(size=60, filled=True).encode(
        x=alt.X('Gameweek:O', title='Gameweek'),
        y=alt.Y('Player_Score:Q', title='Score'),
        color=alt.Color('Result:N', scale=alt.Scale(domain=list(result_color_map.keys()), range=list(result_color_map.values())), title='Result'),
        tooltip=['Player','Gameweek','Player_Score','Result']
    ).properties(
        width=800,
        height=400
    )

    rule = alt.Chart(pd.DataFrame({'y': [line_value]})).mark_rule(color='red').encode(y='y:Q')
    text = alt.Chart(pd.DataFrame({'y': [line_value]})).mark_text(
        align='left', dx=5, dy=-5, color='red'
    ).encode(
        y='y:Q',
        text=alt.value(f"Avg losing score + 1: {line_value:.1f}")
    )

    st.altair_chart((scatter_chart + rule + text), use_container_width=True)

    # Second chart: Player Scores by Gameweek colored by result (original chart)
    second_chart = alt.Chart(chart_data).mark_circle(size=100, filled=True).encode(
        x=alt.X('Player_Score:Q', title='Score'),
        y=alt.Y('Player:N', sort=alt.EncodingSortField(field="Rank", order="ascending"), title='Player'),
        color=alt.Color('Result:N', scale=alt.Scale(domain=list(result_color_map.keys()), range=list(result_color_map.values())), title='Result'),
        tooltip=['Player', 'Gameweek', 'Player_Score', 'Result']
    ).properties(
        width=800,
        height=600,
        title="Player Scores by Gameweek (Colored by Result)"
    ).interactive()

    st.altair_chart(second_chart, use_container_width=True)


    
    

