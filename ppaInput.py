import re
import csv
from datetime import datetime

INPUT_FILE = "ppa_raw.txt"
OUTPUT_FILE = "ppa_matches.csv"


def parse_date(date_str):
    return datetime.strptime(date_str.strip(), "%b %d, %Y").strftime("%Y-%m-%d")


def clean_team(line):
    """Return a tuple of two players, or None if invalid."""
    line = re.sub(r"#\d+\s*", "", line)
    if "/" not in line:
        return None
    parts = line.split("/")
    if len(parts) != 2:
        return None
    return parts[0].strip(), parts[1].strip()


def parse_file():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and l.strip() not in ["Watch", "View"]]

    matches = []
    i = 0

    while i < len(lines):
        # Find tournament header
        if "PPA" in lines[i]:
            tournament = lines[i]

            # Round line
            if i + 1 >= len(lines):
                break
            round_line = lines[i + 1]
            round_name = round_line.split("•")[0].strip()
            date_str = round_line.split("•")[-1].strip()
            date = parse_date(date_str)

            i += 2

            # Skip optional labels
            while i < len(lines) and lines[i] in ["Medal", "Forfeit"]:
                i += 1

            # --- TEAM 1 ---
            team1 = clean_team(lines[i])
            if not team1:
                i += 1
                continue
            team1_p1, team1_p2 = team1
            try:
                team1_sets = int(lines[i + 1])
            except:
                team1_sets = 2

            # Find TEAM 2
            j = i + 2
            while j < len(lines) and not clean_team(lines[j]):
                j += 1
            if j >= len(lines):
                break

            team2 = clean_team(lines[j])
            if not team2:
                i = j + 1
                continue
            team2_p1, team2_p2 = team2
            try:
                team2_sets = int(lines[j + 1])
            except:
                team2_sets = 0

            matches.append([
                tournament,
                round_name,
                date,
                team1_p1,
                team1_p2,
                team2_p1,
                team2_p2,
                team1_sets,
                team2_sets
            ])

            # Move pointer past this match
            i = j + 2
        else:
            i += 1

    return matches



def save_csv(matches):
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tournament",
            "round",
            "date",
            "team1_player1",
            "team1_player2",
            "team2_player1",
            "team2_player2",
            "team1_sets",
            "team2_sets"
        ])
        writer.writerows(matches)


if __name__ == "__main__":
    matches = parse_file()
    save_csv(matches)
    print(f"Saved {len(matches)} matches to {OUTPUT_FILE}")
