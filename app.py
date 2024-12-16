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

st.header("Jimmy Bullardz - ManeBall Analysis")

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
# GAMEWEEK PREVIEWS SECTION
##############################
with st.expander(":soccer: Gameweek Previews :soccer:", expanded=False):
    st.subheader("Gameweek Previews")

    # Load upcoming fixtures
    gameweek_fixtures = pd.read_csv('fixtures.csv')

    # Debugging: Check column names and data
    st.write("Gameweek Fixtures Columns:", gameweek_fixtures.columns)
    st.write(gameweek_fixtures.head())

    # Rename columns to ensure compatibility if necessary
    gameweek_fixtures.columns = gameweek_fixtures.columns.str.strip()  # Remove spaces
    if 'Player1' not in gameweek_fixtures.columns or 'Player2' not in gameweek_fixtures.columns:
        st.error("The fixtures file must have columns named 'Player1' and 'Player2'.")
    else:
        # Calculate probabilities for each match based on player performance
        def calculate_probabilities(player1, player2):
            player1_scores = score_distributions[player1]
            player2_scores = score_distributions[player2]

            total_simulations = 10000
            player1_wins = 0
            player2_wins = 0
            draws = 0

            for _ in range(total_simulations):
                score1 = np.random.choice(player1_scores)
                score2 = np.random.choice(player2_scores)

                if score1 > score2:
                    player1_wins += 1
                elif score2 > score1:
                    player2_wins += 1
                else:
                    draws += 1

            total = player1_wins + player2_wins + draws
            return {
                'Player1': player1,
                'Player2': player2,
                'Player1 Win %': round(100 * player1_wins / total, 1),
                'Player2 Win %': round(100 * player2_wins / total, 1),
                'Draw %': round(100 * draws / total, 1)
            }

        # Generate match predictions
        predictions = []
        for _, row in gameweek_fixtures.iterrows():
            predictions.append(calculate_probabilities(row['Player1'], row['Player2']))

        predictions_df = pd.DataFrame(predictions)

        # Display the predictions
        st.write("Predicted Chances for Upcoming Matches:")
        st.dataframe(predictions_df)

        # Highlight key matchups
        st.write("## Key Matchups to Watch")
        top_matchups = predictions_df.sort_values(by='Draw %', ascending=False).head(3)
        for _, matchup in top_matchups.iterrows():
            st.write(f"- **{matchup['Player1']} vs {matchup['Player2']}**: {matchup['Draw %']}% chance of a draw, {matchup['Player1 Win %']}% chance {matchup['Player1']} wins, {matchup['Player2 Win %']}% chance {matchup['Player2']} wins.")
