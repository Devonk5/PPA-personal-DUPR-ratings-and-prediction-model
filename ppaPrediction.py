import pandas as pd
import os
import math
import difflib

# ====== CONFIG ======
DIVISIONS = {
    '1': {'name': "Men's Doubles",   'match_csv': 'mens_matches.csv',   'elo_csv': 'mens_elo.csv',   'pair_csv': 'mens_pair_elo.csv',   'bet_csv': 'mens_bets.csv'},
    '2': {'name': "Women's Doubles", 'match_csv': 'womens_matches.csv', 'elo_csv': 'womens_elo.csv', 'pair_csv': 'womens_pair_elo.csv', 'bet_csv': 'womens_bets.csv'},
    '3': {'name': "Mixed Doubles",   'match_csv': 'mixed_matches.csv',  'elo_csv': 'mixed_elo.csv',  'pair_csv': 'mixed_pair_elo.csv',  'bet_csv': 'mixed_bets.csv'},
}

INITIAL_ELO = 6
RECENT_MATCHES = 5

# ====== PLAYER ELO DICTIONARY ======
player_elo = {}
recent_elo = {}
matches_played = {}
tournaments_seen = set()

# ====== PAIR ELO DICTIONARY ======
pair_elo = {}
pair_matches = {}
PAIR_MIN_MATCHES = 10
PAIR_WEIGHT = 0.3

def pair_key(p1, p2):
    return tuple(sorted([p1, p2]))

def get_pair_elo(p1, p2):
    key = pair_key(p1, p2)
    if key not in pair_elo:
        pair_elo[key] = (player_elo.get(p1, INITIAL_ELO) + player_elo.get(p2, INITIAL_ELO)) / 2
    return pair_elo[key]

def get_pair_matches(p1, p2):
    return pair_matches.get(pair_key(p1, p2), 0)

# ====== HELPER FUNCTIONS ======
def dynamic_k(team1_elo, team2_elo):
    diff = abs(team1_elo - team2_elo)
    k = 0.04 * (1 + 2 * diff)
    return max(0.02, min(0.12, k))

def get_elo(player):
    return player_elo.get(player, INITIAL_ELO)

def resolve_player(name):
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

def get_dynamic_pair_weight(p1, p2):
    m = get_pair_matches(p1, p2)
    if m < PAIR_MIN_MATCHES:
        return 0.0
    elif m < 30:
        return 0.20
    elif m < 50:
        return 0.30
    elif m < 100:
        return 0.40
    else:
        return 0.50

def team_strength(team):
    p1 = get_effective_elo(team[0])
    p2 = get_effective_elo(team[1])
    individual_strength = 0.6 * max(p1, p2) + 0.4 * min(p1, p2)
    weight = get_dynamic_pair_weight(team[0], team[1])
    if weight > 0:
        pair = get_pair_elo(team[0], team[1])
        return (1 - weight) * individual_strength + weight * pair
    return individual_strength

def get_reliability_score(player):
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
    margin = abs(team1_sets - team2_sets)
    margin_multiplier = 1 + 0.5 * margin
    base_elo_change = k * margin_multiplier * (actual - expected)
    for p in team1:
        rel = get_reliability_score(p) / 100
        k_scale = 0.5 + 0.5 * (1 - rel)
        player_elo[p] = player_elo.get(p, INITIAL_ELO) + base_elo_change * k_scale
        matches_played[p] = matches_played.get(p, 0) + 1
    for p in team2:
        rel = get_reliability_score(p) / 100
        k_scale = 0.5 + 0.5 * (1 - rel)
        player_elo[p] = player_elo.get(p, INITIAL_ELO) - base_elo_change * k_scale
        matches_played[p] = matches_played.get(p, 0) + 1
    key1 = pair_key(team1[0], team1[1])
    key2 = pair_key(team2[0], team2[1])
    pair_elo[key1] = pair_elo.get(key1, (player_elo.get(team1[0], INITIAL_ELO) + player_elo.get(team1[1], INITIAL_ELO)) / 2) + base_elo_change
    pair_elo[key2] = pair_elo.get(key2, (player_elo.get(team2[0], INITIAL_ELO) + player_elo.get(team2[1], INITIAL_ELO)) / 2) - base_elo_change
    pair_matches[key1] = pair_matches.get(key1, 0) + 1
    pair_matches[key2] = pair_matches.get(key2, 0) + 1
    update_recent_form(team1, team2, base_elo_change)

# ====== TRAIN ELO ======
def train_elo(csv_file):
    global player_elo, recent_elo, matches_played, tournaments_seen, pair_elo, pair_matches
    player_elo = {}
    recent_elo = {}
    matches_played = {}
    tournaments_seen = set()
    pair_elo.clear()
    pair_matches.clear()
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

# ====== SAVE PAIR ELO ======
def save_pair_elo(csv_file):
    rows = []
    for (p1, p2), elo in pair_elo.items():
        rows.append({
            'player1': p1,
            'player2': p2,
            'pair_elo': elo,
            'matches_together': pair_matches.get((p1, p2), 0)
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(by='matches_together', ascending=False)
    df.to_csv(csv_file, index=False)
    print(f'Saved pair Elo ratings to {csv_file}')

# ====== LOAD PAIR ELO ======
def load_pair_elo(csv_file):
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            key = pair_key(row['player1'], row['player2'])
            pair_elo[key] = row['pair_elo']
            pair_matches[key] = int(row['matches_together'])
        print(f'Loaded pair Elo ratings from {csv_file}')
    else:
        print('No pair Elo CSV found. Will compute from match history.')

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

# ====== PREDICT MATCH ======
def predict(team1_players, team2_players, scale=0.15):
    team1_elo = team_strength(team1_players)
    team2_elo = team_strength(team2_players)
    diff = team1_elo - team2_elo
    prob_team1_win = 1 / (1 + math.exp(-diff / scale))
    all_players = team1_players + team2_players
    avg_reliability = sum(get_reliability_score(p) for p in all_players) / 400
    uncertainty = 1 - avg_reliability
    prob_team1_win = prob_team1_win * (1 - uncertainty) + 0.5 * uncertainty
    return prob_team1_win

def tournament_accuracy(match_csv, scale=0.375):
    df = pd.read_csv(match_csv)
    df = df.sort_values(by="date").reset_index(drop=True)
    tournaments = df['tournament'].unique()
    results = []
    global player_elo, recent_elo, matches_played
    player_elo = {}
    recent_elo = {}
    pair_elo.clear()
    pair_matches.clear()
    matches_played = {}
    WARMUP_TOURNAMENTS = 11
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
            prob = predict(team1, team2, scale)
            actual = 1 if row['team1_sets'] > row['team2_sets'] else 0
            predicted = 1 if prob > 0.5 else 0
            if predicted == actual:
                correct += 1
            log_loss += -(actual * math.log(prob + 1e-9) + (1 - actual) * math.log(1 - prob + 1e-9))
            total += 1
            update_elo(team1, team2, row['team1_sets'], row['team2_sets'], scale=scale)
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
    global player_elo, recent_elo, matches_played, tournaments_seen, pair_elo, pair_matches
    df = pd.read_csv(match_csv)
    df = df.sort_values(by="date")
    player_elo = {}
    recent_elo = {}
    matches_played = {}
    tournaments_seen = set()
    pair_elo.clear()
    pair_matches.clear()
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
        log_loss += -(actual * math.log(prob + 1e-9) + (1 - actual) * math.log(1 - prob + 1e-9))
        total += 1
        update_elo(team1, team2, row['team1_sets'], row['team2_sets'], scale=scale)
    accuracy = correct / total
    avg_log_loss = log_loss / total
    print(f"Rolling Accuracy: {accuracy:.2%}")
    print(f"Log Loss: {avg_log_loss:.4f}")
    return accuracy, avg_log_loss

# ====== PREDICT MATCH WITH KELLY ======
def predict_match(team1_players, team2_players, bankroll=100, odds_team1=1.8, odds_team2=1.8, scale=0.15, return_kelly=False):
    team1_elo = team_strength(team1_players)
    team2_elo = team_strength(team2_players)
    diff = team1_elo - team2_elo
    prob_team1_win = 1 / (1 + math.exp(-diff / scale))
    all_players = team1_players + team2_players
    avg_reliability = sum(get_reliability_score(p) for p in all_players) / 400
    uncertainty = 1 - avg_reliability
    prob_team1_win = prob_team1_win * (1 - uncertainty) + 0.5 * uncertainty
    prob_team2_win = 1 - prob_team1_win
    result = {"probability_team1": prob_team1_win, "probability_team2": prob_team2_win}
    if return_kelly:
        b1 = odds_team1 - 1
        b2 = odds_team2 - 1
        kelly_team1 = max(0, (b1 * prob_team1_win - prob_team2_win) / b1)
        kelly_team2 = max(0, (b2 * prob_team2_win - prob_team1_win) / b2)
        reliability_factor = avg_reliability
        result.update({
            "suggested_bet_team1": "$" + str(round(bankroll * kelly_team1 * reliability_factor, 2)),
            "suggested_bet_team2": "$" + str(round(bankroll * kelly_team2 * reliability_factor, 2)),
            "reliability_factor": str(round(reliability_factor * 100, 1)) + "%"
        })
    return result

# ====== BET HISTORY ======
def save_bet(csv_file, team1, team2, odds1, odds2, bet_team, bet_amount, prob_team1, prob_team2, reliability_factor, tournament):
    import datetime
    new_row = {
        'date': datetime.date.today().isoformat(),
        'tournament': tournament,
        'team1': team1[0] + ' / ' + team1[1],
        'team2': team2[0] + ' / ' + team2[1],
        'odds_team1': odds1,
        'odds_team2': odds2,
        'prob_team1': round(prob_team1 * 100, 1),
        'prob_team2': round(prob_team2 * 100, 1),
        'reliability_factor': reliability_factor,
        'bet_on': bet_team,
        'bet_amount': bet_amount,
        'result': 'PENDING',
        'pnl': ''
    }
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])
    df.to_csv(csv_file, index=False)
    print(f'Bet saved to {csv_file}')

def view_bet_history(csv_file):
    if not os.path.exists(csv_file):
        print('No bet history found.')
        return
    df = pd.read_csv(csv_file)
    total_bets = len(df)
    settled = df[df['result'] != 'PENDING']
    pending = df[df['result'] == 'PENDING']
    print("\n=== Bet History ===")
    print(df.to_string(index=False))
    print("\nTotal Bets: " + str(total_bets) + " | Settled: " + str(len(settled)) + " | Pending: " + str(len(pending)))
    if len(settled) > 0:
        wins = settled[settled['result'] == 'WIN']
        losses = settled[settled['result'] == 'LOSS']
        pnl = settled['pnl'].apply(lambda x: float(x) if str(x).replace('.','').replace('-','').isdigit() else 0).sum()
        total_staked = settled['bet_amount'].sum()
        roi = (pnl / total_staked * 100) if total_staked > 0 else 0
        print(f'Win/Loss: {len(wins)}W / {len(losses)}L | P&L: ${pnl:.2f} | ROI: {roi:.1f}%')

def settle_bet(csv_file):
    if not os.path.exists(csv_file):
        print('No bet history found.')
        return
    df = pd.read_csv(csv_file)
    pending = df[df['result'] == 'PENDING']
    if pending.empty:
        print('No pending bets to settle.')
        return
    print("\n=== Pending Bets ===")
    for idx, row in pending.iterrows():
        print(f'[{idx}] {row["date"]} | {row["team1"]} vs {row["team2"]} | Bet: ${row["bet_amount"]} on {row["bet_on"]}')
    bet_idx = int(input('Enter bet number to settle: '))
    result = input('Result (WIN/LOSS): ').strip().upper()
    if result not in ['WIN', 'LOSS']:
        print('Invalid result.')
        return
    row = df.loc[bet_idx]
    odds = row['odds_team1'] if row['bet_on'] == row['team1'] else row['odds_team2']
    if result == 'WIN':
        pnl = round(row['bet_amount'] * (odds - 1), 2)
    else:
        pnl = round(-row['bet_amount'], 2)
    df.at[bet_idx, 'result'] = result
    df.at[bet_idx, 'pnl'] = pnl
    df.to_csv(csv_file, index=False)
    print(f'Bet settled: {result} | P&L: ${pnl}')

# ====== MAIN ======
if __name__ == '__main__':
    print("Select division:")
    for k, v in DIVISIONS.items():
        print(f"  {k}. {v['name']}")
    div_choice = input("Division (1/2/3): ").strip()
    if div_choice not in DIVISIONS:
        print("Invalid choice, defaulting to Men's Doubles")
        div_choice = '1'
    cfg = DIVISIONS[div_choice]
    MATCH_CSV    = cfg['match_csv']
    ELO_CSV      = cfg['elo_csv']
    PAIR_ELO_CSV = cfg['pair_csv']
    BET_HISTORY_CSV = cfg['bet_csv']
    print(f"\nLoaded: {cfg['name']}\n")

    while True:
        decision = input("Options: test accuracy(1), accuracy by tournament(2), bet suggestions(3), match predictions(4), Top 10(5), player rating(6), save bet(7), view bet history(8), settle bet(9)\n")
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
                bankroll, odds1, odds2, scale=.15, return_kelly=True
            )
            print(results)
        elif decision == '4':
            players = []
            for i in range(4):
                players.append(resolve_player(input(f"player {i+1}: ")))
            train_elo(MATCH_CSV)
            prob = predict([players[0], players[1]], [players[2], players[3]])
            print(f"\nTeam 1 Win Probability: {prob:.2%}")
            print(f"Team 2 Win Probability: {(1-prob):.2%}\n")
            print(prob)
        elif decision == '5':
            train_elo(MATCH_CSV)
            save_elo(ELO_CSV)
            save_pair_elo(PAIR_ELO_CSV)
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
        elif decision == '7':
            train_elo(MATCH_CSV)
            players = []
            for i in range(4):
                players.append(resolve_player(input(f'player {i+1}: ')))
            tournament = input('Tournament name: ')
            bankroll = float(input('What is our bankroll? '))
            odds1 = float(input('What is the odds for team 1? '))
            odds2 = float(input('What is the odds for team 2? '))
            results = predict_match(
                [players[0], players[1]],
                [players[2], players[3]],
                bankroll, odds1, odds2, scale=.15, return_kelly=True
            )
            print(results)
            bet_team_input = input('Which team did you bet on? (1/2/none): ').strip()
            if bet_team_input in ['1', '2']:
                if bet_team_input == '1':
                    bet_team = players[0] + ' / ' + players[1]
                    bet_amount = float(results['suggested_bet_team1'].replace('$', ''))
                else:
                    bet_team = players[2] + ' / ' + players[3]
                    bet_amount = float(results['suggested_bet_team2'].replace('$', ''))
                confirm = input(f'Save bet of ${bet_amount} on {bet_team}? (y/n): ').strip()
                if confirm == 'y':
                    save_bet(BET_HISTORY_CSV, [players[0], players[1]], [players[2], players[3]],
                             odds1, odds2, bet_team, bet_amount,
                             results['probability_team1'], results['probability_team2'],
                             results['reliability_factor'], tournament)
        elif decision == '8':
            view_bet_history(BET_HISTORY_CSV)
        elif decision == '9':
            settle_bet(BET_HISTORY_CSV)
        else:
            while True:
                leave = input("Do you want to Quit: y/n \n")
                if leave == 'y' or leave == 'n':
                    break
            if leave == 'y':
                break