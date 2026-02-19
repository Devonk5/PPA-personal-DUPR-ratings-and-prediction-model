import pandas as pd
import os
import math

# ====== CONFIG ======
MATCH_CSV = 'ppa_matches.csv'
ELO_CSV = 'player_elo.csv'
INITIAL_ELO = 6
RECENT_MATCHES = 5  # number of recent matches to weigh more heavily

import difflib

# ====== PLAYER ELO DICTIONARY ======
player_elo = {}
recent_elo = {}  # for recent form weighting
matches_played = {}  # track matches played per player
tournaments_seen = set()  # track unique tournaments

# ====== HELPER FUNCTIONS ======
def dynamic_k(team1_elo, team2_elo):
    diff = abs(team1_elo - team2_elo)
    k = 0.04 * (1 + 2 * diff)
    return max(0.02, min(0.12, k))

def get_elo(player):
    return player_elo.get(player, INITIAL_ELO)

def resolve_player(name):
    """Finds the closest matching player name from known players. Returns original if no close match."""
    known = list(player_elo.keys())
    if not known:
        return name
    matches = difflib.get_close_matches(name, known, n=1, cutoff=0.6)
    if matches and matches[0] != name:
        print(f"  [Auto-corrected] '{name}' → '{matches[0]}'")
        return matches[0]
    return name

def get_recent_elo(player):
    history = recent_elo.get(player, [])
    if not history:
        return INITIAL_ELO
    weights = [0.5 ** i for i in range(len(history[-RECENT_MATCHES:]))]
    return sum(h * w for h, w in zip(history[-RECENT_MATCHES:], weights)) / sum(weights)

def get_effective_elo(player):
    base = player_elo.get(player, INITIAL_ELO)
    recent = get_recent_elo(player)
    return 0.7 * recent + 0.3 * base

def team_strength(team):
    p1 = get_effective_elo(team[0])
    p2 = get_effective_elo(team[1])
    return 0.6 * max(p1, p2) + 0.4 * min(p1, p2)

def get_reliability_score(player):
    """
    Returns a reliability score from 0–10 based on matches played relative to the average.
    - 0 matches → 0
    - At average → 5
    - Above average → scales above 5, capped at 10
    - Below average → scales below 5
    """
    played = matches_played.get(player, 0)
    if played == 0:
        return 0.0

    num_tournaments = max(1, len(tournaments_seen))
    threshold = min(30, max(2, num_tournaments // 2))
    all_played = [v for v in matches_played.values() if v >= threshold]
    if not all_played:
        return 0.0

    all_played.sort()
    n = len(all_played)
    median = (all_played[n // 2] if n % 2 != 0
              else (all_played[n // 2 - 1] + all_played[n // 2]) / 2)
    if median == 0:
        return 0.0

    # Scale: score = 5 * (played / median), capped at 20, then multiplied by 5 for 0-100
    score = 5 * (played / median)
    score = min(20.0, max(0.0, score))
    return round(score * 5, 2)

def update_recent_form(team1, team2, base_elo_change):
    for p in team1:
        recent_elo.setdefault(p, [])
        recent_elo[p].append(player_elo[p])
        recent_elo[p] = recent_elo[p][-RECENT_MATCHES:]

    for p in team2:
        recent_elo.setdefault(p, [])
        recent_elo[p].append(player_elo[p])
        recent_elo[p] = recent_elo[p][-RECENT_MATCHES:]

def update_elo(team1, team2, team1_sets, team2_sets, scale=0.1):
    team1_elos = [get_effective_elo(p) for p in team1]
    team2_elos = [get_effective_elo(p) for p in team2]

    team1_strength = 0.6 * max(team1_elos) + 0.4 * min(team1_elos)
    team2_strength = 0.6 * max(team2_elos) + 0.4 * min(team2_elos)

    expected = 1 / (1 + math.exp(-(team1_strength - team2_strength) / scale))
    actual = 1 if team1_sets > team2_sets else 0

    k = dynamic_k(team1_strength, team2_strength)

    # ---- MARGIN MULTIPLIER HERE ----
    margin = abs(team1_sets - team2_sets)
    margin_multiplier = 1 + 0.5 * margin

    base_elo_change = k * margin_multiplier * (actual - expected)

    for p in team1:
        rel = get_reliability_score(p) / 100
        k_scale = 0.5 + 0.5 * (1 - rel)  # unreliable players update faster (more uncertain)
        player_elo[p] = player_elo.get(p, INITIAL_ELO) + base_elo_change * k_scale
        matches_played[p] = matches_played.get(p, 0) + 1
    for p in team2:
        rel = get_reliability_score(p) / 100
        k_scale = 0.5 + 0.5 * (1 - rel)
        player_elo[p] = player_elo.get(p, INITIAL_ELO) - base_elo_change * k_scale
        matches_played[p] = matches_played.get(p, 0) + 1

    update_recent_form(team1, team2, base_elo_change)


# ====== TRAIN ELO ======
def train_elo(csv_file):
    global tournaments_seen
    df = pd.read_csv(csv_file)
    df = df.sort_values(by="date")
    for _, row in df.iterrows():
        team1 = [row['team1_player1'], row['team1_player2']]
        team2 = [row['team2_player1'], row['team2_player2']]
        if 'tournament' in row:
            tournaments_seen.add(row['tournament'])
        update_elo(team1, team2, row['team1_sets'], row['team2_sets'])

# ====== SAVE ELO ======
def save_elo(csv_file):
    rows = []
    for player, elo in player_elo.items():
        rows.append({
            'player': player,
            'elo': elo,
            'matches_played': matches_played.get(player, 0),
            'reliability_score': get_reliability_score(player)
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(by='elo', ascending=False)
    df.to_csv(csv_file, index=False)
    print(f"Saved Elo ratings to {csv_file}")

# ====== LOAD ELO ======
def load_elo(csv_file):
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            player_elo[row['player']] = row['elo']
            if 'matches_played' in df.columns:
                matches_played[row['player']] = int(row['matches_played'])
        print(f"Loaded Elo ratings from {csv_file}")
    else:
        print("No Elo CSV found. Will compute from match history.")

def test_scales(match_csv, scales):
    results = []

    for scale in scales:
        print(f"\nTesting scale = {scale}")

        # Reset Elo
        global player_elo, recent_elo, matches_played, tournaments_seen
        player_elo = {}
        recent_elo = {}
        matches_played = {}

        df = pd.read_csv(match_csv)
        df = df.sort_values(by="date").reset_index(drop=True)

        correct = 0
        total = 0
        log_loss = 0

        for _, row in df.iterrows():
            team1 = [row['team1_player1'], row['team1_player2']]
            team2 = [row['team2_player1'], row['team2_player2']]

            prob = predict(team1, team2, scale)
            actual = 1 if row['team1_sets'] > row['team2_sets'] else 0

            predicted = 1 if prob > 0.5 else 0
            if predicted == actual:
                correct += 1

            log_loss += -(actual * math.log(prob + 1e-9) +
                          (1 - actual) * math.log(1 - prob + 1e-9))

            total += 1

            update_elo(team1, team2,
                       row['team1_sets'], row['team2_sets'],
                       scale=scale)

        accuracy = correct / total
        avg_log_loss = log_loss / total

        results.append((scale, accuracy, avg_log_loss))

        print(f"Accuracy: {accuracy:.4f}, Log Loss: {avg_log_loss:.4f}")

    return results


# ====== PREDICT MATCH ======
def predict(team1_players, team2_players, scale=0.15):
    team1_elo = team_strength(team1_players)
    team2_elo = team_strength(team2_players)
    diff = team1_elo - team2_elo
    prob_team1_win = 1 / (1 + math.exp(-diff / scale))
    # Nudge probability toward 50/50 based on unreliability
    all_players = team1_players + team2_players
    avg_reliability = sum(get_reliability_score(p) for p in all_players) / 400  # 0-1 scale
    uncertainty = 1 - avg_reliability
    prob_team1_win = prob_team1_win * (1 - uncertainty) + 0.5 * uncertainty
    return prob_team1_win

def tournament_accuracy(match_csv, scale=0.375):
    df = pd.read_csv(match_csv)
    df = df.sort_values(by="date").reset_index(drop=True)

    tournaments = df['tournament'].unique()
    results = []

    # Reset Elo at the start of the season
    global player_elo, recent_elo, matches_played
    player_elo = {}
    recent_elo = {}
    matches_played = {}

    WARMUP_TOURNAMENTS = 5

    # Track cumulative metrics across tournaments (post-warmup only)
    cum_correct = 0
    cum_total = 0
    cum_log_loss = 0

    for t_idx, t in enumerate(tournaments):
        t_matches = df[df['tournament'] == t]
        is_warmup = t_idx < WARMUP_TOURNAMENTS

        correct = 0
        total = 0
        log_loss = 0

        for _, row in t_matches.iterrows():
            team1 = [row['team1_player1'], row['team1_player2']]
            team2 = [row['team2_player1'], row['team2_player2']]

            # Predict BEFORE updating Elo
            prob = predict(team1, team2, scale)
            actual = 1 if row['team1_sets'] > row['team2_sets'] else 0

            predicted = 1 if prob > 0.5 else 0
            if predicted == actual:
                correct += 1
            log_loss += -(actual * math.log(prob + 1e-9) + (1 - actual) * math.log(1 - prob + 1e-9))
            total += 1

            # --- Update Elo AFTER prediction ---
            update_elo(team1, team2, row['team1_sets'], row['team2_sets'], scale=scale)

        # Tournament metrics
        accuracy = correct / total
        avg_log_loss = log_loss / total
        results.append((t, accuracy, avg_log_loss))

        if is_warmup:
            print(f"Tournament: {t} → [WARMUP] Accuracy: {accuracy:.2%}, Log Loss: {avg_log_loss:.4f}")
        else:
            print(f"Tournament: {t} → Accuracy: {accuracy:.2%}, Log Loss: {avg_log_loss:.4f}")
            cum_correct += correct
            cum_total += total
            cum_log_loss += log_loss
            cum_accuracy = cum_correct / cum_total
            cum_avg_log_loss = cum_log_loss / cum_total
            print(f"Cumulative Accuracy (post-warmup): {cum_accuracy:.2%}, Log Loss: {cum_avg_log_loss:.4f}")
        print()

    if cum_total > 0:
        print(f"=== Final Post-Warmup Accuracy: {cum_correct / cum_total:.2%} ===")
        print(f"=== Final Post-Warmup Log Loss: {cum_log_loss / cum_total:.4f} ===")

    return results



# ====== ROLLING EVALUATION ======
def compute_accuracy(match_csv, scale=0.1):
    global player_elo, recent_elo, matches_played, tournaments_seen

    df = pd.read_csv(match_csv)
    df = df.sort_values(by="date")

    player_elo = {}
    recent_elo = {}
    matches_played = {}
    tournaments_seen = set()

    correct = 0
    total = 0
    log_loss = 0

    for _, row in df.iterrows():
        team1 = [row['team1_player1'], row['team1_player2']]
        team2 = [row['team2_player1'], row['team2_player2']]

        # Predict BEFORE updating
        prob = predict(team1, team2, scale)

        actual = 1 if row['team1_sets'] > row['team2_sets'] else 0

        # Accuracy
        predicted = 1 if prob > 0.5 else 0
        if predicted == actual:
            correct += 1

        # Log Loss
        log_loss += -(actual * math.log(prob + 1e-9) +
                      (1 - actual) * math.log(1 - prob + 1e-9))
        total += 1

        # Update AFTER prediction
        update_elo(team1, team2, row['team1_sets'], row['team2_sets'], scale=scale)

    accuracy = correct / total
    avg_log_loss = log_loss / total

    print(f"Rolling Accuracy: {accuracy:.2%}")
    print(f"Log Loss: {avg_log_loss:.4f}")

    return accuracy, avg_log_loss

# ====== PREDICT MATCH WITH KELLY ======
def predict_match(team1_players, team2_players, bankroll=100, odds_team1=1.8, odds_team2=1.8, scale=0.15, return_kelly=False):
    """
    Predicts the probability of Team 1 winning and optionally calculates Kelly stakes.

    Parameters:
    - team1_players: list of 2 player names
    - team2_players: list of 2 player names
    - bankroll: total bankroll (used if return_kelly is True)
    - odds_team1: decimal odds offered by bookmaker for Team 1
    - odds_team2: decimal odds offered by bookmaker for Team 2
    - scale: Elo scale factor for logistic probability
    - return_kelly: if True, returns recommended stake fraction for both teams

    Returns:
    - dict with keys:
        probability: model probability of Team 1 winning
        (optional) kelly_team1: fraction of bankroll to bet on Team 1
        (optional) kelly_team2: fraction of bankroll to bet on Team 2
        (optional) suggested_bet_team1: bankroll * kelly_team1
        (optional) suggested_bet_team2: bankroll * kelly_team2
    """

    # Compute team strengths
    team1_elo = team_strength(team1_players)
    team2_elo = team_strength(team2_players)

    # Logistic probability of Team 1 winning
    diff = team1_elo - team2_elo
    prob_team1_win = 1 / (1 + math.exp(-diff / scale))
    # Nudge probability toward 50/50 based on unreliability
    all_players = team1_players + team2_players
    avg_reliability = sum(get_reliability_score(p) for p in all_players) / 400  # 0-1 scale
    uncertainty = 1 - avg_reliability
    prob_team1_win = prob_team1_win * (1 - uncertainty) + 0.5 * uncertainty
    prob_team2_win = 1 - prob_team1_win

    result = {"probability_team1": prob_team1_win}
    result.update(
        {"probability_team2": prob_team2_win}
        )

    if return_kelly:
        # Kelly fraction formula: f* = (b * p - q) / b
        b1 = odds_team1 - 1
        b2 = odds_team2 - 1

        kelly_team1 = max(0, (b1 * prob_team1_win - prob_team2_win) / b1)
        kelly_team2 = max(0, (b2 * prob_team2_win - prob_team1_win) / b2)

        # Scale bets by average reliability of all 4 players (already computed above)
        reliability_factor = avg_reliability  # 0 = no bet, 1 = full Kelly

        result.update({
            "suggested_bet_team1": "$" + str(round(bankroll * kelly_team1 * reliability_factor, 2)),
            "suggested_bet_team2": "$" + str(round(bankroll * kelly_team2 * reliability_factor, 2)),
            "reliability_factor": str(round(reliability_factor * 100, 1)) + "%"
        })


    return result


# ====== MAIN ======
if __name__ == '__main__':
    while True:
        decision = input("Options: test accuracy(1), test accuracy by tournament(2) Bet suggestions(3), match predictions(4), Top 10(5), get player rating(6)\n")
        if decision == '1':
            compute_accuracy(MATCH_CSV, 0.1)
        elif decision == '2':
            tournament_accuracy(MATCH_CSV, .15)
        elif decision == '3':
            train_elo(MATCH_CSV)
            players = []
            for i in range(4):
                players.append(resolve_player(input(f"player {i+1}: ")))
            bankroll = float(input("What is our bankroll? "))
            odds1 = float(input("What is the odds for team 1? "))
            odds2 = float(input("What is the odds for team 2? "))

            results = predict_match(
                [players[0], players[1]],
                [players[2], players[3]],
                bankroll,
                odds1,
                odds2,
                scale=.15,
                return_kelly=True
            )
            print(results)

        elif decision == '4':
            players = []
            for i in range(4):
                players.append(resolve_player(input(f"player {i+1}: ")))
            train_elo(MATCH_CSV)
            prob = predict([players[0], players[1]],
               [players[2], players[3]])
            player_elo = {}
            recent_elo = {}
            matches_played = {}

            print(f"\nTeam 1 Win Probability: {prob:.2%}")
            print(f"Team 2 Win Probability: {(1-prob):.2%}\n")
            print(prob)

        elif decision == '5':
            train_elo(MATCH_CSV)
            save_elo(ELO_CSV)
            if not player_elo:
                print("Elo ratings not computed yet. Run accuracy test or process matches first.")
            else:
                top_players = sorted(player_elo.items(), key=lambda x: x[1], reverse=True)[:10]
                print("\n=== Top 10 Players by Elo ===")
                for rank, (player, elo) in enumerate(top_players, start=1):
                    played = matches_played.get(player, 0)
                    reliability = get_reliability_score(player)
                    print(f"{rank}. {player}: {elo:.2f} | Matches: {played} | Reliability: {reliability}%")
                print()

        elif decision == '6':
            train_elo(MATCH_CSV)
            player = input("Who's Rating are you looking for?\n")
            played = matches_played.get(player, 0)
            reliability = get_reliability_score(player)
            print(f"{player}: ELO={get_elo(player):.2f} | Matches Played={played} | Reliability={reliability}%")
        else:
            
            while True:
                leave = input("Do you want to Quit: y/n \n")
                if(leave == 'y' or leave == 'n'):
                    break
            if(leave == 'y'):
                break