# PPA-personal-DUPR-ratings-and-prediction-model
dataset of ppa matches and player evaluations to make match up predictions
PPA Pickleball ELO Prediction System
A doubles pickleball ELO rating and match prediction system built for PPA tour data, featuring reliability-weighted betting suggestions using the Kelly Criterion.

Requirements

Python 3.8+
pandas

bashpip install pandas

Files
FileDescriptionppaPrediction.pyMain scriptppa_matches.csvMatch history (required)player_elo.csvSaved ELO ratings (auto-generated)

Match Data Format
ppa_matches.csv must have the following columns:
ColumnDescriptiondateMatch date (used for chronological ordering)tournamentTournament nameteam1_player1Player 1 on Team 1team1_player2Player 2 on Team 1team2_player1Player 1 on Team 2team2_player2Player 2 on Team 2team1_setsSets won by Team 1team2_setsSets won by Team 2

Usage
Run the script:
bashpy ppaPrediction.py
You will be presented with a menu:
Options: test accuracy(1), test accuracy by tournament(2), Bet suggestions(3), match predictions(4), Top 10(5), get player rating(6)
Option 1 — Rolling Accuracy
Tests overall model accuracy across all matches in chronological order using a rolling ELO update.
Option 2 — Accuracy by Tournament
Shows accuracy per tournament with cumulative tracking. The first 5 tournaments are treated as warmup and excluded from the final accuracy score to account for the cold-start period where ELO ratings have little data. Prints a final post-warmup accuracy at the end.
Option 3 — Bet Suggestions
Enter 4 player names, your bankroll, and bookmaker odds to receive Kelly Criterion bet sizing recommendations. Bets are scaled by the average reliability of all 4 players — low-reliability matchups result in smaller suggested bets.
Option 4 — Match Prediction
Enter 4 player names to get a win probability for each team.
Option 5 — Top 10 Players
Displays the top 10 players by ELO rating, along with their matches played and reliability score.
Option 6 — Player Rating
Look up a specific player's ELO, matches played, and reliability score.

How ELO Works
Base ELO

All players start at an initial ELO of 6
ELO updates after every match based on expected vs actual outcome

Effective ELO
Each player's effective ELO is a blend of their base ELO and recent form:
effective_elo = 0.7 * recent_elo + 0.3 * base_elo
Recent form weights the last 5 matches with exponential decay.
Team Strength
team_strength = 0.6 * max(player_elos) + 0.4 * min(player_elos)
The stronger player on a team contributes more to the team's overall strength.
Dynamic K Factor
The K factor (how much ELO changes per match) scales with the ELO difference between teams and is capped between 0.02 and 0.12. A margin multiplier also amplifies changes for lopsided scorelines:
margin_multiplier = 1 + 0.5 * set_margin
Reliability-Scaled K Factor
Players with low reliability (few matches) have a higher K factor, meaning their ELO updates faster since there is more uncertainty about their true skill. Reliable veterans update more conservatively.

Reliability Score
Each player receives a reliability score from 0% to 100% based on matches played relative to the median of qualifying players.
How it's calculated

Only players with at least min(30, num_tournaments // 2) matches are used to compute the median — this threshold grows dynamically as your dataset grows, capping at 30
reliability = 5 * (matches_played / median) * 5 → scaled to 0–100%
0 matches → 0%
At median → 50%
Above median scales proportionally higher

How reliability affects the system
Bet sizing: Kelly Criterion stakes are multiplied by the average reliability of all 4 players:
suggested_bet = kelly_stake * avg_reliability
A matchup with all 4 players at 100% reliability gets the full Kelly bet. A matchup with unknown players gets a heavily reduced bet.
Prediction confidence: Win probabilities are nudged toward 50/50 based on average unreliability:
adjusted_prob = raw_prob * avg_reliability + 0.5 * (1 - avg_reliability)
This prevents overconfident predictions for players with thin data.

Kelly Criterion
The Kelly Criterion calculates the optimal fraction of your bankroll to bet:
f* = (b * p - q) / b
Where b = decimal odds - 1, p = model win probability, q = 1 - p.
Kelly bets are then scaled by the reliability factor before being suggested.

Player Name Matching
Player names are fuzzy-matched using Python's difflib — minor typos are automatically corrected with a notification:
[Auto-corrected] 'Stacksrud F.' → 'Staksrud F.'

Accuracy Expectations
AccuracyAssessment< 55%Below baseline, model needs tuning55–60%Decent, some signal present60–65%Good, comparable to most sports ELO models65–70%Very good70%+Excellent
Target post-warmup accuracy: 63–67%

Configuration
At the top of ppaPrediction.py:
VariableDefaultDescriptionMATCH_CSVppa_matches.csvPath to match dataELO_CSVplayer_elo.csvPath to save ELO ratingsINITIAL_ELO6Starting ELO for all playersRECENT_MATCHES5Number of recent matches for form weighting
