import re
import csv
from datetime import datetime

INPUT_FILE = "ppa_raw.txt"

OUTPUT_FILES = {
    'mens':         'mens_matches.csv',
    'womens':       'womens_matches.csv',
    'mixed':        'mixed_matches.csv',
    'mens_singles': 'mens_singles_matches.csv',
    'womens_singles': 'womens_singles_matches.csv',
}

TOURNAMENT_KEYWORDS = ["PPA", "UPA", "MLP", "APP"]

HEADERS = ["tournament", "round", "date", "team1_player1", "team1_player2",
           "team2_player1", "team2_player2", "team1_sets", "team2_sets"]


def parse_date(date_str):
    return datetime.strptime(date_str.strip(), "%b %d, %Y").strftime("%Y-%m-%d")


def is_tournament_header(line):
    return any(keyword in line for keyword in TOURNAMENT_KEYWORDS)


def get_division(round_line):
    """Extract division from round line e.g. 'Finals • Mens Doubles • Nov 10, 2024'"""
    line_lower = round_line.lower()
    if 'mixed' in line_lower:
        return 'mixed'
    elif 'women' in line_lower:
        if 'singles' in line_lower:
            return 'womens_singles'
        return 'womens'
    elif 'men' in line_lower:
        if 'singles' in line_lower:
            return 'mens_singles'
        return 'mens'
    return None  # unknown - skip


def clean_team(line):
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

    matches = {'mens': [], 'womens': [], 'mixed': [], 'mens_singles': [], 'womens_singles': []}
    skipped = 0
    i = 0

    while i < len(lines):
        if is_tournament_header(lines[i]):
            tournament = lines[i]

            if i + 1 >= len(lines):
                break
            round_line = lines[i + 1]
            round_name = round_line.split("•")[0].strip()
            date_str = round_line.split("•")[-1].strip()

            try:
                date = parse_date(date_str)
            except ValueError:
                i += 1
                continue

            division = get_division(round_line)
            if division is None:
                skipped += 1
                i += 2
                continue  # truly unknown format

            i += 2

            while i < len(lines) and lines[i] in ["Medal", "Forfeit"]:
                i += 1

            if i >= len(lines):
                break
            team1 = clean_team(lines[i])
            if not team1:
                i += 1
                continue
            team1_p1, team1_p2 = team1
            try:
                team1_sets = int(lines[i + 1])
            except:
                team1_sets = 2

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

            matches[division].append([
                tournament, round_name, date,
                team1_p1, team1_p2,
                team2_p1, team2_p2,
                team1_sets, team2_sets
            ])

            i = j + 2
        else:
            i += 1

    return matches, skipped


def save_csvs(matches):
    for division, rows in matches.items():
        if not rows:
            print(f"  No matches found for {division} — skipping")
            continue
        filepath = OUTPUT_FILES[division]
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(HEADERS)
            writer.writerows(rows)
        print(f"  Saved {len(rows):>4} matches to {filepath}")


if __name__ == "__main__":
    matches, skipped = parse_file()
    total = sum(len(v) for v in matches.values())
    print(f"\nParsed {total} doubles matches ({skipped} singles/unknown skipped)\n")
    save_csvs(matches)