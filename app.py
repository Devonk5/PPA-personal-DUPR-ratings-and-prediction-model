from flask import Flask, request, jsonify, render_template_string
import sys
import os
import io
import pandas as pd

# Import everything from ppaPrediction
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppaPrediction as elo_module
from ppaPrediction import (
    train_elo, predict, predict_match, resolve_player,
    save_bet, get_reliability_score, get_elo,
)

# ====== DIVISION CONFIG ======
DIVISIONS = {
    'mens':   {'match_csv': 'mens_matches.csv',   'elo_csv': 'mens_elo.csv',   'pair_csv': 'mens_pair_elo.csv',   'bet_csv': 'mens_bets.csv'},
    'womens': {'match_csv': 'womens_matches.csv', 'elo_csv': 'womens_elo.csv', 'pair_csv': 'womens_pair_elo.csv', 'bet_csv': 'womens_bets.csv'},
    'mixed':  {'match_csv': 'mixed_matches.csv',  'elo_csv': 'mixed_elo.csv',  'pair_csv': 'mixed_pair_elo.csv',  'bet_csv': 'mixed_bets.csv'},
}

def get_csvs(division):
    return DIVISIONS.get(division, DIVISIONS['mens'])

app = Flask(__name__)

HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PPA Pickleball ELO</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #080c10;
    --surface: #0e1419;
    --surface2: #141c24;
    --border: #1e2d3d;
    --accent: #00d4ff;
    --accent2: #ff6b35;
    --green: #00ff9d;
    --red: #ff3d5a;
    --text: #e8f4f8;
    --muted: #4a6278;
    --glow: 0 0 20px rgba(0,212,255,0.15);
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Animated grid background */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
  }

  .container {
    max-width: 1100px;
    margin: 0 auto;
    padding: 0 24px;
    position: relative;
    z-index: 1;
  }

  /* Header */
  header {
    padding: 32px 0 24px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 32px;
  }

  .header-inner {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .logo {
    display: flex;
    align-items: baseline;
    gap: 12px;
  }

  .logo-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.4rem;
    letter-spacing: 3px;
    color: var(--accent);
    text-shadow: 0 0 30px rgba(0,212,255,0.4);
  }

  .logo-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
  }

  .status-badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    padding: 6px 14px;
    border: 1px solid var(--green);
    color: var(--green);
    border-radius: 2px;
    letter-spacing: 2px;
    text-transform: uppercase;
    animation: pulse-border 2s ease-in-out infinite;
  }

  @keyframes pulse-border {
    0%, 100% { box-shadow: 0 0 8px rgba(0,255,157,0.2); }
    50% { box-shadow: 0 0 16px rgba(0,255,157,0.5); }
  }

  /* Nav tabs */
  .nav-tabs {
    display: flex;
    gap: 4px;
    margin-bottom: 32px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0;
    overflow-x: auto;
  }

  .tab {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 10px 20px;
    cursor: pointer;
    color: var(--muted);
    border: none;
    background: none;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
    white-space: nowrap;
    margin-bottom: -1px;
  }

  .tab:hover { color: var(--text); }
  .tab.active {
    color: var(--accent);
    border-bottom-color: var(--accent);
  }

  /* Panels */
  .panel { display: none; animation: fadeIn 0.3s ease; }
  .panel.active { display: block; }

  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
  }

  /* Cards */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 28px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
  }

  .card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    opacity: 0.4;
  }

  .card-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.1rem;
    letter-spacing: 2px;
    color: var(--accent);
    margin-bottom: 20px;
  }

  /* Form elements */
  .form-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 20px;
  }

  .form-grid-4 {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1fr;
    gap: 12px;
    margin-bottom: 20px;
  }

  .form-group { display: flex; flex-direction: column; gap: 6px; }

  label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
  }

  input {
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 10px 14px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    border-radius: 3px;
    transition: border-color 0.2s, box-shadow 0.2s;
    outline: none;
    width: 100%;
  }

  input:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 2px rgba(0,212,255,0.1);
  }

  input::placeholder { color: var(--muted); }

  /* Buttons */
  .btn {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 12px 28px;
    border: none;
    border-radius: 3px;
    cursor: pointer;
    transition: all 0.2s;
  }

  .btn-primary {
    background: var(--accent);
    color: var(--bg);
    font-weight: 500;
  }

  .btn-primary:hover {
    background: #33ddff;
    box-shadow: 0 0 20px rgba(0,212,255,0.3);
  }

  .btn-secondary {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--muted);
  }

  .btn-secondary:hover {
    border-color: var(--accent);
    color: var(--accent);
  }

  .btn-danger {
    background: transparent;
    border: 1px solid var(--red);
    color: var(--red);
  }

  .btn-danger:hover { background: rgba(255,61,90,0.1); }

  .btn:disabled { opacity: 0.4; cursor: not-allowed; }

  /* Result box */
  .result-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 3px;
    padding: 20px;
    margin-top: 20px;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.8;
    display: none;
  }

  .result-box.show { display: block; animation: fadeIn 0.3s ease; }
  .result-box.error { border-left-color: var(--red); }

  /* Probability display */
  .prob-display {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-top: 20px;
  }

  .prob-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
  }

  .prob-card.winner { border-color: var(--green); }
  .prob-card.loser { border-color: var(--border); opacity: 0.7; }

  .prob-team {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 1px;
    margin-bottom: 8px;
  }

  .prob-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.5rem;
    line-height: 1;
    margin-bottom: 4px;
  }

  .prob-card.winner .prob-value { color: var(--green); text-shadow: 0 0 20px rgba(0,255,157,0.3); }
  .prob-card.loser .prob-value { color: var(--muted); }

  .prob-bar {
    height: 3px;
    background: var(--border);
    border-radius: 2px;
    margin-top: 12px;
    overflow: hidden;
  }

  .prob-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
  }

  .prob-card.winner .prob-fill { background: var(--green); }
  .prob-card.loser .prob-fill { background: var(--muted); }

  /* Kelly display */
  .kelly-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 12px;
    margin-top: 16px;
  }

  .kelly-stat {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 14px;
    text-align: center;
  }

  .kelly-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: var(--muted);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 6px;
  }

  .kelly-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.6rem;
    color: var(--accent2);
  }

  /* Top players table */
  .players-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
  }

  .players-table th {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }

  .players-table td {
    padding: 12px 14px;
    border-bottom: 1px solid rgba(30,45,61,0.5);
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
  }

  .players-table tr:hover td { background: var(--surface2); }

  .rank-num {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.1rem;
    color: var(--muted);
  }

  .player-name { color: var(--text); font-weight: 500; }

  .elo-value {
    color: var(--accent);
    font-weight: 500;
  }

  .reliability-bar {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .rel-track {
    flex: 1;
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
    max-width: 80px;
  }

  .rel-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent2), var(--green));
    border-radius: 2px;
  }

  /* Accuracy stats */
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 24px;
  }

  .stat-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 20px;
    text-align: center;
  }

  .stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
  }

  .stat-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.4rem;
    color: var(--green);
  }

  /* Bet history table */
  .bet-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.78rem;
    overflow-x: auto;
  }

  .bet-table th {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--muted);
    padding: 8px 12px;
    text-align: left;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }

  .bet-table td {
    padding: 10px 12px;
    border-bottom: 1px solid rgba(30,45,61,0.5);
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    white-space: nowrap;
  }

  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 2px;
    font-size: 0.6rem;
    letter-spacing: 1px;
    font-weight: 500;
  }

  .badge-pending { background: rgba(74,98,120,0.3); color: var(--muted); border: 1px solid var(--border); }
  .badge-win { background: rgba(0,255,157,0.1); color: var(--green); border: 1px solid rgba(0,255,157,0.3); }
  .badge-loss { background: rgba(255,61,90,0.1); color: var(--red); border: 1px solid rgba(255,61,90,0.3); }

  .pnl-pos { color: var(--green); }
  .pnl-neg { color: var(--red); }

  /* Loading spinner */
  .spinner {
    display: inline-block;
    width: 14px; height: 14px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    margin-right: 8px;
    vertical-align: middle;
  }

  @keyframes spin { to { transform: rotate(360deg); } }

  .loading { display: none; align-items: center; color: var(--muted); font-family: 'DM Mono', monospace; font-size: 0.75rem; margin-top: 16px; }
  .loading.show { display: flex; }

  /* Settle modal */
  .modal-overlay {
    position: fixed; inset: 0;
    background: rgba(8,12,16,0.85);
    backdrop-filter: blur(4px);
    z-index: 100;
    display: none;
    align-items: center;
    justify-content: center;
  }

  .modal-overlay.show { display: flex; }

  .modal {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 32px;
    width: 480px;
    max-width: 90vw;
    position: relative;
  }

  .modal-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.3rem;
    letter-spacing: 2px;
    color: var(--accent);
    margin-bottom: 20px;
  }

  .modal-close {
    position: absolute;
    top: 16px; right: 16px;
    background: none; border: none;
    color: var(--muted);
    font-size: 1.2rem;
    cursor: pointer;
    line-height: 1;
  }

  .modal-close:hover { color: var(--text); }

  .result-buttons { display: flex; gap: 12px; margin-top: 20px; }

  /* Scrollable output */
  .output-scroll {
    max-height: 400px;
    overflow-y: auto;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    line-height: 1.9;
    color: var(--text);
  }

  .output-scroll::-webkit-scrollbar { width: 4px; }
  .output-scroll::-webkit-scrollbar-track { background: var(--surface2); }
  .output-scroll::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  .t-warmup { color: var(--muted); }
  .t-name { color: var(--accent); }
  .t-acc { color: var(--green); }
  .t-loss { color: var(--accent2); }
  .t-cum { color: var(--text); opacity: 0.6; font-size: 0.7rem; }
  .t-final { color: var(--green); font-size: 0.85rem; margin-top: 8px; }

  /* Division selector */
  .division-bar {
    display: flex;
    gap: 8px;
    margin-bottom: 24px;
    flex-wrap: wrap;
  }

  .div-btn {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 7px 16px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--muted);
    border-radius: 2px;
    cursor: pointer;
    transition: all 0.2s;
  }

  .div-btn:hover { border-color: var(--accent); color: var(--accent); }

  .div-btn.active {
    background: var(--accent);
    border-color: var(--accent);
    color: var(--bg);
    font-weight: 500;
  }

  .div-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 1px;
    padding: 3px 8px;
    border-radius: 2px;
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.3);
    color: var(--accent);
    margin-left: 10px;
    vertical-align: middle;
  }

  @media (max-width: 640px) {
    .form-grid, .form-grid-4 { grid-template-columns: 1fr 1fr; }
    .prob-display { grid-template-columns: 1fr; }
    .stats-grid { grid-template-columns: 1fr; }
    .kelly-grid { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="header-inner">
      <div class="logo">
        <span class="logo-title">PPA ELO</span>
        <span class="logo-sub">Pickleball Analytics</span>
      </div>
      <span class="status-badge">● Live</span>
    </div>
  </header>

  <div class="division-bar">
    <button class="div-btn active" onclick="switchDivision('mens', this)">Men's Doubles</button>
    <button class="div-btn" onclick="switchDivision('womens', this)">Women's Doubles</button>
    <button class="div-btn" onclick="switchDivision('mixed', this)">Mixed Doubles</button>
  </div>

  <nav class="nav-tabs">
    <button class="tab active" onclick="switchTab('predict')">Predict</button>
    <button class="tab" onclick="switchTab('bet')">Bet Advisor</button>
    <button class="tab" onclick="switchTab('history')">Bet History</button>
    <button class="tab" onclick="switchTab('rankings')">Rankings</button>
    <button class="tab" onclick="switchTab('teams')">Team ELO</button>
    <button class="tab" onclick="switchTab('accuracy')">Accuracy</button>
    <button class="tab" onclick="switchTab('player')">Player Lookup</button>
  </nav>

  <!-- PREDICT -->
  <div id="panel-predict" class="panel active">
    <div class="card">
      <div class="card-title">Match Prediction <span class="div-badge" id="predict-div-badge">Men's Doubles</span></div>
      <div class="form-grid-4">
        <div class="form-group"><label>Team 1 — Player A</label><input id="p1" placeholder="e.g. Johns B." /></div>
        <div class="form-group"><label>Team 1 — Player B</label><input id="p2" placeholder="e.g. Newman R." /></div>
        <div class="form-group"><label>Team 2 — Player A</label><input id="p3" placeholder="e.g. Staksrud F." /></div>
        <div class="form-group"><label>Team 2 — Player B</label><input id="p4" placeholder="e.g. Tellez P." /></div>
      </div>
      <button class="btn btn-primary" onclick="runPredict()">Run Prediction</button>
      <div class="loading" id="predict-loading"><span class="spinner"></span>Calculating...</div>
      <div id="predict-result"></div>
    </div>
  </div>

  <!-- BET ADVISOR -->
  <div id="panel-bet" class="panel">
    <div class="card">
      <div class="card-title">Kelly Bet Advisor <span class="div-badge" id="bet-div-badge">Men's Doubles</span></div>
      <div class="form-grid-4">
        <div class="form-group"><label>Team 1 — Player A</label><input id="b-p1" placeholder="e.g. Johns B." /></div>
        <div class="form-group"><label>Team 1 — Player B</label><input id="b-p2" placeholder="e.g. Newman R." /></div>
        <div class="form-group"><label>Team 2 — Player A</label><input id="b-p3" placeholder="e.g. Staksrud F." /></div>
        <div class="form-group"><label>Team 2 — Player B</label><input id="b-p4" placeholder="e.g. Tellez P." /></div>
      </div>
      <div class="form-grid">
        <div class="form-group"><label>Bankroll ($)</label><input id="b-bankroll" type="number" placeholder="1000" /></div>
        <div class="form-group"><label>Tournament</label><input id="b-tournament" placeholder="e.g. 2025 PPA Masters" /></div>
        <div class="form-group"><label>Team 1 Odds (decimal)</label><input id="b-odds1" type="number" step="0.01" placeholder="1.90" /></div>
        <div class="form-group"><label>Team 2 Odds (decimal)</label><input id="b-odds2" type="number" step="0.01" placeholder="1.90" /></div>
      </div>
      <div style="display:flex;gap:12px;">
        <button class="btn btn-primary" onclick="runBet()">Calculate</button>
      </div>
      <div class="loading" id="bet-loading"><span class="spinner"></span>Calculating...</div>
      <div id="bet-result"></div>
    </div>
  </div>

  <!-- TEAM ELO -->
  <div id="panel-teams" class="panel">
    <div class="card">
      <div class="card-title">Top 10 Pairs <span class="div-badge" id="teams-div-badge">Men's Doubles</span></div>
      <button class="btn btn-primary" onclick="loadTeams()" style="margin-bottom:20px;">Load Team Rankings</button>
      <div class="loading" id="teams-loading"><span class="spinner"></span>Training model...</div>
      <div id="teams-result"></div>
    </div>
  </div>

  <!-- BET HISTORY -->
  <div id="panel-history" class="panel">
    <div class="card">
      <div class="card-title">Bet History <span class="div-badge" id="history-div-badge">Men's Doubles</span></div>
      <div style="display:flex;gap:12px;margin-bottom:20px;">
        <button class="btn btn-primary" onclick="loadHistory()">Refresh</button>
      </div>
      <div id="history-result"><div style="color:var(--muted);font-family:DM Mono,monospace;font-size:0.8rem;">Click refresh to load bet history.</div></div>
    </div>
  </div>

  <!-- RANKINGS -->
  <div id="panel-rankings" class="panel">
    <div class="card">
      <div class="card-title">Top 10 Players <span class="div-badge" id="rankings-div-badge">Men's Doubles</span></div>
      <button class="btn btn-primary" onclick="loadRankings()" style="margin-bottom:20px;">Load Rankings</button>
      <div class="loading" id="rankings-loading"><span class="spinner"></span>Training model...</div>
      <div id="rankings-result"></div>
    </div>
  </div>

  <!-- ACCURACY -->
  <div id="panel-accuracy" class="panel">
    <div class="card">
      <div class="card-title">Tournament Accuracy <span class="div-badge" id="accuracy-div-badge">Men's Doubles</span></div>
      <p style="color:var(--muted);font-size:0.82rem;margin-bottom:20px;font-family:DM Mono,monospace;">Runs tournament-by-tournament accuracy test with 8-tournament warmup period.</p>
      <button class="btn btn-primary" onclick="runAccuracy()">Run Accuracy Test</button>
      <div class="loading" id="accuracy-loading"><span class="spinner"></span>Testing all tournaments...</div>
      <div id="accuracy-result"></div>
    </div>
  </div>

  <!-- PLAYER LOOKUP -->
  <div id="panel-player" class="panel">
    <div class="card">
      <div class="card-title">Player Lookup <span class="div-badge" id="player-div-badge">Men's Doubles</span></div>
      <div style="display:flex;gap:12px;align-items:flex-end;">
        <div class="form-group" style="flex:1"><label>Player Name</label><input id="pl-name" placeholder="e.g. Johns B." onkeydown="if(event.key==='Enter')lookupPlayer()" /></div>
        <button class="btn btn-primary" onclick="lookupPlayer()">Search</button>
      </div>
      <div class="loading" id="player-loading"><span class="spinner"></span>Looking up...</div>
      <div id="player-result"></div>
    </div>
  </div>
</div>

<!-- Settle Modal -->
<div class="modal-overlay" id="settle-modal">
  <div class="modal">
    <button class="modal-close" onclick="closeSettle()">✕</button>
    <div class="modal-title">Settle Bet</div>
    <div id="settle-info" style="font-family:DM Mono,monospace;font-size:0.8rem;color:var(--muted);margin-bottom:16px;line-height:1.7;"></div>
    <div class="result-buttons">
      <button class="btn btn-primary" onclick="settleBet('WIN')" style="background:var(--green);flex:1;">WIN</button>
      <button class="btn btn-danger" onclick="settleBet('LOSS')" style="flex:1;">LOSS</button>
    </div>
  </div>
</div>

<!-- Save Bet Modal -->
<div class="modal-overlay" id="save-modal">
  <div class="modal">
    <button class="modal-close" onclick="closeSave()">✕</button>
    <div class="modal-title">Save Bet</div>
    <div id="save-info" style="font-family:DM Mono,monospace;font-size:0.8rem;color:var(--muted);margin-bottom:16px;line-height:1.7;"></div>
    <div style="display:flex;gap:12px;margin-bottom:16px;">
      <div style="flex:1;text-align:center;">
        <div style="font-family:DM Mono,monospace;font-size:0.62rem;color:var(--muted);letter-spacing:1px;margin-bottom:6px;">BET ON</div>
        <button class="btn btn-secondary" id="save-t1-btn" onclick="selectTeam(1)" style="width:100%;">Team 1</button>
      </div>
      <div style="flex:1;text-align:center;">
        <div style="font-family:DM Mono,monospace;font-size:0.62rem;color:var(--muted);letter-spacing:1px;margin-bottom:6px;">BET ON</div>
        <button class="btn btn-secondary" id="save-t2-btn" onclick="selectTeam(2)" style="width:100%;">Team 2</button>
      </div>
    </div>
    <button class="btn btn-primary" onclick="confirmSave()" style="width:100%;">Save Bet</button>
  </div>
</div>

<script>
let currentBetData = null;
let currentSettleIdx = null;
let selectedTeam = null;
let currentDivision = 'mens';

function switchTab(name) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
  event.target.classList.add('active');
}

function switchDivision(div, btn) {
  currentDivision = div;
  document.querySelectorAll('.div-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  const labels = {'mens': "Men's Doubles", 'womens': "Women's Doubles", 'mixed': 'Mixed Doubles'};
  const label = labels[div] || div;
  ['predict','bet','history','rankings','teams','accuracy','player'].forEach(id => {
    const badge = document.getElementById(id + '-div-badge');
    if (badge) badge.textContent = label;
    const res = document.getElementById(id + '-result');
    if (res) res.innerHTML = '';
  });
}

async function api(endpoint, data) {
  const res = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
  return res.json();
}

function showLoading(id) { document.getElementById(id).classList.add('show'); }
function hideLoading(id) { document.getElementById(id).classList.remove('show'); }

async function runPredict() {
  const players = [
    document.getElementById('p1').value.trim(),
    document.getElementById('p2').value.trim(),
    document.getElementById('p3').value.trim(),
    document.getElementById('p4').value.trim()
  ];
  if (players.some(p => !p)) return alert('Please enter all 4 players.');
  showLoading('predict-loading');
  document.getElementById('predict-result').innerHTML = '';
  const data = await api('/api/predict', { players, division: currentDivision });
  hideLoading('predict-loading');
  const p1 = (data.prob_team1 * 100).toFixed(1);
  const p2 = (data.prob_team2 * 100).toFixed(1);
  const t1wins = data.prob_team1 > data.prob_team2;
  document.getElementById('predict-result').innerHTML = `
    <div class="prob-display">
      <div class="prob-card ${t1wins ? 'winner' : 'loser'}">
        <div class="prob-team">${data.team1[0]} / ${data.team1[1]}</div>
        <div class="prob-value">${p1}%</div>
        <div style="font-family:DM Mono,monospace;font-size:0.62rem;color:var(--muted);">WIN PROBABILITY</div>
        <div class="prob-bar"><div class="prob-fill" style="width:${p1}%"></div></div>
      </div>
      <div class="prob-card ${!t1wins ? 'winner' : 'loser'}">
        <div class="prob-team">${data.team2[0]} / ${data.team2[1]}</div>
        <div class="prob-value">${p2}%</div>
        <div style="font-family:DM Mono,monospace;font-size:0.62rem;color:var(--muted);">WIN PROBABILITY</div>
        <div class="prob-bar"><div class="prob-fill" style="width:${p2}%"></div></div>
      </div>
    </div>
    ${data.corrected ? `<div style="font-family:DM Mono,monospace;font-size:0.68rem;color:var(--muted);margin-top:12px;">⚡ ${data.corrected.join(' · ')}</div>` : ''}
  `;
}

async function runBet() {
  const players = [
    document.getElementById('b-p1').value.trim(),
    document.getElementById('b-p2').value.trim(),
    document.getElementById('b-p3').value.trim(),
    document.getElementById('b-p4').value.trim()
  ];
  const bankroll = parseFloat(document.getElementById('b-bankroll').value);
  const odds1 = parseFloat(document.getElementById('b-odds1').value);
  const odds2 = parseFloat(document.getElementById('b-odds2').value);
  const tournament = document.getElementById('b-tournament').value.trim();
  if (players.some(p => !p) || !bankroll || !odds1 || !odds2) return alert('Please fill all fields.');
  showLoading('bet-loading');
  document.getElementById('bet-result').innerHTML = '';
  const data = await api('/api/bet', { players, bankroll, odds1, odds2, tournament, division: currentDivision });
  hideLoading('bet-loading');
  currentBetData = data;

  const p1 = (data.prob_team1 * 100).toFixed(1);
  const p2 = (data.prob_team2 * 100).toFixed(1);
  const t1wins = data.prob_team1 > data.prob_team2;

  document.getElementById('bet-result').innerHTML = `
    <div class="prob-display" style="margin-top:20px;">
      <div class="prob-card ${t1wins ? 'winner' : 'loser'}">
        <div class="prob-team">${data.team1[0]} / ${data.team1[1]}</div>
        <div class="prob-value">${p1}%</div>
        <div class="prob-bar"><div class="prob-fill" style="width:${p1}%"></div></div>
      </div>
      <div class="prob-card ${!t1wins ? 'winner' : 'loser'}">
        <div class="prob-team">${data.team2[0]} / ${data.team2[1]}</div>
        <div class="prob-value">${p2}%</div>
        <div class="prob-bar"><div class="prob-fill" style="width:${p2}%"></div></div>
      </div>
    </div>
    <div class="kelly-grid">
      <div class="kelly-stat">
        <div class="kelly-label">Bet on Team 1</div>
        <div class="kelly-value">${data.bet_team1}</div>
      </div>
      <div class="kelly-stat">
        <div class="kelly-label">Bet on Team 2</div>
        <div class="kelly-value">${data.bet_team2}</div>
      </div>
      <div class="kelly-stat">
        <div class="kelly-label">Reliability</div>
        <div class="kelly-value" style="color:var(--accent)">${data.reliability}</div>
      </div>
    </div>
    <button class="btn btn-secondary" onclick="openSave()" style="margin-top:16px;">Save Bet →</button>
    ${data.corrected ? `<div style="font-family:DM Mono,monospace;font-size:0.68rem;color:var(--muted);margin-top:12px;">⚡ ${data.corrected.join(' · ')}</div>` : ''}
  `;
}

function openSave() {
  if (!currentBetData) return;
  selectedTeam = null;
  document.getElementById('save-t1-btn').textContent = currentBetData.team1[0] + ' / ' + currentBetData.team1[1];
  document.getElementById('save-t2-btn').textContent = currentBetData.team2[0] + ' / ' + currentBetData.team2[1];
  document.getElementById('save-t1-btn').style.borderColor = '';
  document.getElementById('save-t2-btn').style.borderColor = '';
  document.getElementById('save-info').innerHTML =
    `Tournament: ${currentBetData.tournament || 'N/A'}<br>
     Team 1: ${currentBetData.team1.join(' / ')} — ${currentBetData.bet_team1}<br>
     Team 2: ${currentBetData.team2.join(' / ')} — ${currentBetData.bet_team2}`;
  document.getElementById('save-modal').classList.add('show');
}

function closeSave() { document.getElementById('save-modal').classList.remove('show'); }

function selectTeam(n) {
  selectedTeam = n;
  document.getElementById('save-t1-btn').style.borderColor = n === 1 ? 'var(--accent)' : '';
  document.getElementById('save-t2-btn').style.borderColor = n === 2 ? 'var(--accent)' : '';
}

async function confirmSave() {
  if (!selectedTeam) return alert('Select a team to bet on.');
  const d = currentBetData;
  const betTeam = selectedTeam === 1 ? d.team1.join(' / ') : d.team2.join(' / ');
  const betAmount = selectedTeam === 1
    ? parseFloat(d.bet_team1.replace('$',''))
    : parseFloat(d.bet_team2.replace('$',''));
  await api('/api/save_bet', { division: currentDivision,
    team1: d.team1, team2: d.team2,
    odds1: d.odds1, odds2: d.odds2,
    bet_team: betTeam, bet_amount: betAmount,
    prob_team1: d.prob_team1, prob_team2: d.prob_team2,
    reliability_factor: d.reliability,
    tournament: d.tournament
  });
  closeSave();
  alert('Bet saved!');
}

async function loadHistory() {
  const data = await api('/api/history', { division: currentDivision });
  const el = document.getElementById('history-result');
  if (!data.bets || data.bets.length === 0) {
    el.innerHTML = '<div style="color:var(--muted);font-family:DM Mono,monospace;font-size:0.8rem;">No bets found.</div>';
    return;
  }

  let statsHtml = '';
  if (data.stats) {
    const s = data.stats;
    statsHtml = `<div class="stats-grid" style="grid-template-columns:repeat(4,1fr)">
      <div class="stat-card"><div class="stat-label">Total Bets</div><div class="stat-value">${s.total}</div></div>
      <div class="stat-card"><div class="stat-label">Win Rate</div><div class="stat-value" style="color:var(--green)">${s.win_rate}</div></div>
      <div class="stat-card"><div class="stat-label">P&L</div><div class="stat-value" style="color:${parseFloat(s.pnl) >= 0 ? 'var(--green)' : 'var(--red)'}">${s.pnl}</div></div>
      <div class="stat-card"><div class="stat-label">ROI</div><div class="stat-value" style="color:${parseFloat(s.roi) >= 0 ? 'var(--green)' : 'var(--red)'}">${s.roi}</div></div>
    </div>`;
  }

  const rows = data.bets.map((b, i) => `
    <tr>
      <td>${b.date}</td>
      <td style="color:var(--text)">${b.team1}</td>
      <td>vs</td>
      <td style="color:var(--text)">${b.team2}</td>
      <td style="color:var(--accent)">${b.bet_on}</td>
      <td>$${b.bet_amount}</td>
      <td><span class="badge badge-${b.result.toLowerCase()}">${b.result}</span></td>
      <td class="${b.pnl > 0 ? 'pnl-pos' : b.pnl < 0 ? 'pnl-neg' : ''}">${b.pnl !== '' ? '$'+b.pnl : '—'}</td>
      ${b.result === 'PENDING' ? `<td><button class="btn btn-secondary" style="padding:4px 10px;font-size:0.6rem" onclick="openSettle(${b.idx}, '${b.team1}', '${b.team2}', '${b.bet_on}', ${b.bet_amount})">Settle</button></td>` : '<td></td>'}
    </tr>`).join('');

  el.innerHTML = statsHtml + `
    <div style="overflow-x:auto">
    <table class="bet-table">
      <thead><tr>
        <th>Date</th><th>Team 1</th><th></th><th>Team 2</th><th>Bet On</th><th>Amount</th><th>Result</th><th>P&L</th><th></th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table></div>`;
}

function openSettle(idx, team1, team2, betOn, amount) {
  currentSettleIdx = idx;
  document.getElementById('settle-info').innerHTML =
    `${team1} vs ${team2}<br>Bet: $${amount} on ${betOn}`;
  document.getElementById('settle-modal').classList.add('show');
}

function closeSettle() { document.getElementById('settle-modal').classList.remove('show'); }

async function settleBet(result) {
  await api('/api/settle', { idx: currentSettleIdx, result, division: currentDivision });
  closeSettle();
  loadHistory();
}

async function loadRankings() {
  showLoading('rankings-loading');
  document.getElementById('rankings-result').innerHTML = '';
  const data = await api('/api/rankings', { division: currentDivision });
  hideLoading('rankings-loading');
  const rows = data.players.map((p, i) => `
    <tr>
      <td><span class="rank-num">${i+1}</span></td>
      <td class="player-name">${p.name}</td>
      <td class="elo-value">${p.elo}</td>
      <td>${p.matches}</td>
      <td>
        <div class="reliability-bar">
          <div class="rel-track"><div class="rel-fill" style="width:${p.reliability}%"></div></div>
          <span style="font-family:DM Mono,monospace;font-size:0.7rem;color:var(--muted)">${p.reliability}%</span>
        </div>
      </td>
    </tr>`).join('');

  document.getElementById('rankings-result').innerHTML = `
    <table class="players-table">
      <thead><tr><th>#</th><th>Player</th><th>ELO</th><th>Matches</th><th>Reliability</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

async function loadTeams() {
  showLoading('teams-loading');
  document.getElementById('teams-result').innerHTML = '';
  const data = await api('/api/teams', { division: currentDivision });
  hideLoading('teams-loading');
  if (data.error) {
    document.getElementById('teams-result').innerHTML = `<div class="result-box error show">${data.error}</div>`;
    return;
  }
  const rows = data.pairs.map((p, i) => `
    <tr>
      <td><span class="rank-num">${i+1}</span></td>
      <td class="player-name">${p.player1}</td>
      <td style="color:var(--muted);font-family:DM Mono,monospace;font-size:0.7rem;">+</td>
      <td class="player-name">${p.player2}</td>
      <td class="elo-value">${p.pair_elo}</td>
      <td style="font-family:DM Mono,monospace;font-size:0.8rem;color:var(--muted)">${p.matches}</td>
    </tr>`).join('');
  document.getElementById('teams-result').innerHTML = `
    <table class="players-table">
      <thead><tr><th>#</th><th>Player 1</th><th></th><th>Player 2</th><th>Pair ELO</th><th>Matches</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

async function runAccuracy() {
  showLoading('accuracy-loading');
  document.getElementById('accuracy-result').innerHTML = '';
  const data = await api('/api/accuracy', { division: currentDivision });
  hideLoading('accuracy-loading');

  const lines = data.results.map(r => {
    if (r.warmup) {
      return `<div class="t-warmup">[WARMUP] <span class="t-name">${r.tournament}</span> — ${(r.accuracy*100).toFixed(1)}%</div>`;
    }
    return `<div>
      <span class="t-name">${r.tournament}</span>
      <span class="t-acc"> ${(r.accuracy*100).toFixed(1)}%</span>
      <span style="color:var(--muted)"> · Loss: </span><span class="t-loss">${r.log_loss.toFixed(4)}</span>
    </div>
    <div class="t-cum">↳ Cumulative: ${(r.cum_accuracy*100).toFixed(2)}%</div>`;
  }).join('');

  document.getElementById('accuracy-result').innerHTML = `
    <div class="stats-grid" style="margin-top:20px;">
      <div class="stat-card"><div class="stat-label">Final Accuracy</div><div class="stat-value">${(data.final_accuracy*100).toFixed(2)}%</div></div>
      <div class="stat-card"><div class="stat-label">Log Loss</div><div class="stat-value" style="color:var(--accent2)">${data.final_log_loss.toFixed(4)}</div></div>
      <div class="stat-card"><div class="stat-label">Tournaments</div><div class="stat-value" style="color:var(--accent)">${data.results.filter(r=>!r.warmup).length}</div></div>
    </div>
    <div class="card" style="margin-top:0">
      <div class="output-scroll">${lines}</div>
    </div>`;
}

async function lookupPlayer() {
  const name = document.getElementById('pl-name').value.trim();
  if (!name) return;
  showLoading('player-loading');
  document.getElementById('player-result').innerHTML = '';
  const data = await api('/api/player', { name, division: currentDivision });
  hideLoading('player-loading');

  if (data.error) {
    document.getElementById('player-result').innerHTML =
      `<div class="result-box error show">${data.error}</div>`;
    return;
  }

  const relWidth = Math.min(100, data.reliability);
  document.getElementById('player-result').innerHTML = `
    <div style="margin-top:20px;display:grid;grid-template-columns:repeat(3,1fr);gap:16px;">
      <div class="stat-card"><div class="stat-label">Player</div><div style="font-family:Bebas Neue,sans-serif;font-size:1.6rem;color:var(--text)">${data.name}</div></div>
      <div class="stat-card"><div class="stat-label">ELO Rating</div><div class="stat-value">${data.elo}</div></div>
      <div class="stat-card"><div class="stat-label">Matches Played</div><div class="stat-value" style="color:var(--accent)">${data.matches}</div></div>
    </div>
    <div class="card" style="margin-top:16px;">
      <div class="stat-label" style="margin-bottom:12px;">Reliability Score</div>
      <div style="display:flex;align-items:center;gap:16px;">
        <div style="flex:1;height:8px;background:var(--border);border-radius:4px;overflow:hidden;">
          <div style="width:${relWidth}%;height:100%;background:linear-gradient(90deg,var(--accent2),var(--green));border-radius:4px;transition:width 1s ease;"></div>
        </div>
        <span style="font-family:Bebas Neue,sans-serif;font-size:1.8rem;color:var(--green)">${data.reliability}%</span>
      </div>
    </div>
    ${data.corrected ? `<div style="font-family:DM Mono,monospace;font-size:0.7rem;color:var(--muted);margin-top:8px;">⚡ Auto-corrected: "${data.original}" → "${data.name}"</div>` : ''}
  `;
}
</script>
</body>
</html>'''

@app.route('/')
def index():
    return render_template_string(HTML)

def _train(division='mens'):
    cfg = get_csvs(division)
    if not os.path.exists(cfg['match_csv']):
        raise FileNotFoundError(f"Match CSV not found: {cfg['match_csv']}")
    elo_module.train_elo(cfg['match_csv'])

@app.route('/api/predict', methods=['POST'])
def api_predict():
    d = request.json
    div = d.get('division', 'mens')
    try:
        _train(div)
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    corrected = []
    players = []
    for p in d['players']:
        r = elo_module.resolve_player(p)
        if r != p:
            corrected.append(f"'{p}' → '{r}'")
        players.append(r)
    prob = elo_module.predict([players[0], players[1]], [players[2], players[3]])
    return jsonify({
        'prob_team1': prob,
        'prob_team2': 1 - prob,
        'team1': [players[0], players[1]],
        'team2': [players[2], players[3]],
        'corrected': corrected if corrected else None
    })

@app.route('/api/bet', methods=['POST'])
def api_bet():
    d = request.json
    div = d.get('division', 'mens')
    try:
        _train(div)
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    corrected = []
    players = []
    for p in d['players']:
        r = elo_module.resolve_player(p)
        if r != p:
            corrected.append(f"'{p}' → '{r}'")
        players.append(r)
    result = elo_module.predict_match(
        [players[0], players[1]], [players[2], players[3]],
        bankroll=d['bankroll'], odds_team1=d['odds1'], odds_team2=d['odds2'],
        scale=0.15, return_kelly=True
    )
    return jsonify({
        'prob_team1': result['probability_team1'],
        'prob_team2': result['probability_team2'],
        'bet_team1': result['suggested_bet_team1'],
        'bet_team2': result['suggested_bet_team2'],
        'reliability': result['reliability_factor'],
        'team1': [players[0], players[1]],
        'team2': [players[2], players[3]],
        'odds1': d['odds1'],
        'odds2': d['odds2'],
        'tournament': d.get('tournament', ''),
        'corrected': corrected if corrected else None
    })

@app.route('/api/save_bet', methods=['POST'])
def api_save_bet():
    d = request.json
    cfg = get_csvs(d.get('division', 'mens'))
    elo_module.save_bet(
        cfg['bet_csv'],
        d['team1'], d['team2'],
        d['odds1'], d['odds2'],
        d['bet_team'], d['bet_amount'],
        d['prob_team1'], d['prob_team2'],
        d['reliability_factor'], d['tournament']
    )
    return jsonify({'ok': True})

@app.route('/api/history', methods=['POST'])
def api_history():
    import pandas as pd
    d = request.json
    cfg = get_csvs(d.get('division', 'mens'))
    if not os.path.exists(cfg['bet_csv']):
        return jsonify({'bets': [], 'stats': None})
    df = pd.read_csv(cfg['bet_csv'])
    bets = []
    for idx, row in df.iterrows():
        bets.append({
            'idx': int(idx),
            'date': str(row['date']),
            'tournament': str(row.get('tournament', '')),
            'team1': str(row['team1']),
            'team2': str(row['team2']),
            'bet_on': str(row['bet_on']),
            'bet_amount': float(row['bet_amount']),
            'result': str(row['result']),
            'pnl': float(row['pnl']) if str(row['pnl']) not in ['', 'nan'] else ''
        })
    settled = df[df['result'] != 'PENDING']
    stats = None
    if len(settled) > 0:
        wins = len(settled[settled['result'] == 'WIN'])
        losses = len(settled[settled['result'] == 'LOSS'])
        pnl = settled['pnl'].apply(lambda x: float(x) if str(x).replace('.','').replace('-','').isdigit() else 0).sum()
        staked = settled['bet_amount'].sum()
        roi = (pnl / staked * 100) if staked > 0 else 0
        stats = {
            'total': len(df),
            'win_rate': f"{(wins/(wins+losses)*100):.1f}%" if (wins+losses) > 0 else "N/A",
            'pnl': f"{'+'if pnl>=0 else ''}{pnl:.2f}",
            'roi': f"{'+'if roi>=0 else ''}{roi:.1f}%"
        }
    return jsonify({'bets': bets, 'stats': stats})

@app.route('/api/settle', methods=['POST'])
def api_settle():
    d = request.json
    import pandas as pd
    cfg = get_csvs(d.get('division', 'mens'))
    df = pd.read_csv(cfg['bet_csv'])
    idx = d['idx']
    result = d['result']
    row = df.loc[idx]
    odds = row['odds_team1'] if row['bet_on'] == row['team1'] else row['odds_team2']
    pnl = round(row['bet_amount'] * (odds - 1), 2) if result == 'WIN' else round(-row['bet_amount'], 2)
    df.at[idx, 'result'] = result
    df.at[idx, 'pnl'] = pnl
    df.to_csv(cfg['bet_csv'], index=False)
    return jsonify({'ok': True, 'pnl': pnl})

@app.route('/api/rankings', methods=['POST'])
def api_rankings():
    d = request.json
    div = d.get('division', 'mens')
    try:
        _train(div)
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    top = sorted(elo_module.player_elo.items(), key=lambda x: x[1], reverse=True)[:10]
    players = []
    for name, elo in top:
        players.append({
            'name': name,
            'elo': round(elo, 3),
            'matches': elo_module.matches_played.get(name, 0),
            'reliability': elo_module.get_reliability_score(name)
        })
    return jsonify({'players': players})

@app.route('/api/accuracy', methods=['POST'])
def api_accuracy():
    import io, sys
    d = request.json
    cfg = get_csvs(d.get('division', 'mens'))
    if not os.path.exists(cfg['match_csv']):
        return jsonify({'error': f"Match CSV not found: {cfg['match_csv']}"}), 404

    # Capture stdout from tournament_accuracy
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        raw_results = elo_module.tournament_accuracy(cfg['match_csv'], scale=0.15)
    finally:
        sys.stdout = old_stdout

    # Parse results into structured format
    results = []
    cum_correct = cum_total = cum_log_loss = 0
    WARMUP = 11
    for t_idx, (t, acc, ll) in enumerate(raw_results):
        is_warmup = t_idx < WARMUP
        entry = {'tournament': t, 'accuracy': acc, 'log_loss': ll, 'warmup': is_warmup}
        if not is_warmup:
            # rough match count from accuracy - use raw_results as-is
            entry['cum_accuracy'] = acc  # will be overwritten below
            entry['cum_log_loss'] = ll
        else:
            entry['cum_accuracy'] = 0
            entry['cum_log_loss'] = 0
        results.append(entry)

    # Recompute cumulative properly
    cum_correct = cum_total = 0
    cum_log_loss = 0.0
    for i, (t, acc, ll) in enumerate(raw_results):
        if i >= WARMUP:
            # We don't have match counts from tournament_accuracy return value
            # so use the captured output to get final numbers
            pass

    # Get final numbers from the captured buffer output
    output = buffer.getvalue()
    final_acc = 0
    final_ll = 0
    for line in output.split('\n'):
        if 'Final Post-Warmup Accuracy' in line:
            try:
                final_acc = float(line.split(':')[1].strip().replace('%','')) / 100
            except:
                pass
        if 'Final Post-Warmup Log Loss' in line:
            try:
                final_ll = float(line.split(':')[1].strip())
            except:
                pass

    # Recompute cumulative from scratch using same logic as tournament_accuracy
    import pandas as pd, math as _math
    df = pd.read_csv(cfg['match_csv'])
    df = df.sort_values(by='date').reset_index(drop=True)
    tournaments = df['tournament'].unique()
    elo_module.player_elo = {}
    elo_module.recent_elo = {}
    elo_module.matches_played = {}
    elo_module.tournaments_seen = set()
    elo_module.pair_elo.clear()
    elo_module.pair_matches.clear()
    cum_correct = cum_total = 0
    cum_log_loss_total = 0.0
    results2 = []
    for t_idx, t in enumerate(tournaments):
        t_matches = df[df['tournament'] == t]
        is_warmup = t_idx < WARMUP
        correct = total = 0
        log_loss = 0.0
        for _, row in t_matches.iterrows():
            team1 = [row['team1_player1'], row['team1_player2']]
            team2 = [row['team2_player1'], row['team2_player2']]
            prob = elo_module.predict(team1, team2, 0.15)
            actual = 1 if row['team1_sets'] > row['team2_sets'] else 0
            if (1 if prob > 0.5 else 0) == actual:
                correct += 1
            log_loss += -(_math.log(prob + 1e-9) if actual else _math.log(1 - prob + 1e-9))
            total += 1
            elo_module.update_elo(team1, team2, row['team1_sets'], row['team2_sets'], scale=0.15)
        acc = correct / total
        ll = log_loss / total
        entry = {'tournament': t, 'accuracy': acc, 'log_loss': ll, 'warmup': is_warmup}
        if not is_warmup:
            cum_correct += correct
            cum_total += total
            cum_log_loss_total += log_loss
            entry['cum_accuracy'] = cum_correct / cum_total
            entry['cum_log_loss'] = cum_log_loss_total / cum_total
        else:
            entry['cum_accuracy'] = 0
            entry['cum_log_loss'] = 0
        results2.append(entry)

    final_acc2 = cum_correct / cum_total if cum_total > 0 else 0
    final_ll2 = cum_log_loss_total / cum_total if cum_total > 0 else 0
    return jsonify({'results': results2, 'final_accuracy': final_acc2, 'final_log_loss': final_ll2})

@app.route('/api/teams', methods=['POST'])
def api_teams():
    d = request.json
    div = d.get('division', 'mens')
    try:
        _train(div)
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    pairs = []
    for (p1, p2), elo in elo_module.pair_elo.items():
        matches = elo_module.pair_matches.get((p1, p2), 0)
        pairs.append({'player1': p1, 'player2': p2, 'pair_elo': round(elo, 3), 'matches': matches})
    pairs = sorted(pairs, key=lambda x: x['pair_elo'], reverse=True)[:10]
    return jsonify({'pairs': pairs})

@app.route('/api/player', methods=['POST'])
def api_player():
    d = request.json
    div = d.get('division', 'mens')
    try:
        _train(div)
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    name = d['name']
    resolved = elo_module.resolve_player(name)
    if resolved not in elo_module.player_elo:
        return jsonify({'error': f'Player "{name}" not found.'})
    return jsonify({
        'name': resolved,
        'original': name,
        'corrected': resolved != name,
        'elo': round(elo_module.get_elo(resolved), 3),
        'matches': elo_module.matches_played.get(resolved, 0),
        'reliability': elo_module.get_reliability_score(resolved)
    })

if __name__ == '__main__':
    print("Starting PPA ELO server at http://localhost:5000")
    app.run(debug=True, port=5000)