import urllib.request
from bs4 import BeautifulSoup
from models import LeagueGame


def construct_url_for_split(league, split, year):
    url = "https://lol.gamepedia.com/" + league + "/" + str(year) \
      + "_Season/" + split + "_Season/Picks_and_Bans"
    return url


def extract_table_rows_from_url(url):
    fp = urllib.request.urlopen(url)
    page_bytes = fp.read()
    page_html = page_bytes.decode("utf8")
    fp.close()
    soup = BeautifulSoup(page_html, 'html.parser')
    draft_table = soup.find_all('table', id="pbh-table")[0]
    table_rows = draft_table.find_all('tr')[2:]
    return table_rows


def extract_game_from_row(row_of_interest, verbose=True):

    table_columns = row_of_interest.find_all('td')
    blue = table_columns[1]

    blue_win = False
    if blue.get("class") is not None and blue.get("class")[0] == "pbh-winner":
        blue_win = True

    patch = table_columns[4].get_text()

    champs_columns = table_columns[11:15] + table_columns[19:22]

    # bp1, rp1-2, bp2-3, rp3, rp4, bp4-5, rp5

    non_champ_classes = {'pbh-blue', 'pbh-cell', 'pbh-red', 'pbh-divider'}

    blue_champs = set()
    red_champs = set()

    for i in range(len(champs_columns)):
        column = champs_columns[i]
        classes = column.get("class")
        for column_class in classes:
            if column_class not in non_champ_classes:
                champ = column_class.split("-")[1]
                if i in {0, 2, 5}:
                    blue_champs.add(champ)
                else:
                    red_champs.add(champ)

    game = LeagueGame(blue_champs, red_champs, blue_win, patch)

    if verbose:
        print("game blue champs: ", game.blue_champs)
        print("game red champs: ", game.red_champs)
        print("game blue win: ", game.blue_win)
        print("game patch: ", game.patch)
        print()

    return game


if __name__ == "__main__":
    # scrap every row
    url = construct_url_for_split("LCK", "Summer", 2018)
    table_rows = extract_table_rows_from_url(url)
    for row in table_rows:
        extract_game_from_row(row, True)
        # set this to false if you dont want to see prints
