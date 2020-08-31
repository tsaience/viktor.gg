import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from models import DuoInfo, MatchupInfo
from utils import process_champ_name, export_pickle, read_pickle


chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("window-size=1920,1080")
chromedriver_path = "/Users/tsaience/chromedriver"  # SET THIS UP FOR YOUR COMPUTER
browser = webdriver.Chrome(
    executable_path=chromedriver_path, options=chrome_options)

# master matchup dict, {(champ, role): {(matchup, role): matchupinfo}}
# master duo dict, {(champ, role): {(duo, role): duoinfo}}}

roles_url = {
    "middle": "mid-lane-tier-list",
    "top": "top-lane-tier-list",
    "support": "support-tier-list",
    "adc": "adc-tier-list",
    "jungle": "jungle-tier-list"
}


def scrap_all_champs():
    url_all_champs = "https://u.gg/lol/champions"
    browser.get(url_all_champs)
    champ_names_table = \
        browser.find_element_by_class_name(
            "champions-container").get_attribute("innerHTML")
    champ_soup = BeautifulSoup(champ_names_table, "html.parser")
    champ_names_elements = champ_soup.find_all(
        "div", {"class": "champion-name"})
    all_champs = [process_champ_name(i.text) for i in champ_names_elements]
    return all_champs


def scrap_role_champs(role):
    # role can be "top", "jungle", "middle", "adc", "support"
    url = "https://u.gg/lol/" + roles_url[role]
    browser.get(url)
    for _ in range(3):
        time.sleep(1)
        scroll_js = "window.scrollTo(0, document.body.scrollHeight); \
                     var lenOfPage=document.body.scrollHeight;"
        browser.execute_script(scroll_js)
    tier_list_table = browser.find_element_by_class_name(
        "rt-tbody").get_attribute("innerHTML")
    tier_list_soup = BeautifulSoup(tier_list_table, "html.parser")
    rows = tier_list_soup.find_all("strong", {"class": "champion-name"})
    champs_name = [process_champ_name(i.text) for i in rows]
    return champs_name


# scrap matchups for a champion in a given role
def scrap_champ_matchups(champ_name, role):
    url = "https://u.gg/lol/champions/" + champ_name + "/matchups?role=" + role
    browser.get(url)
    print(url)
    matchups_table = WebDriverWait(browser, 10).until(
        EC.presence_of_element_located(
            (By.CLASS_NAME, "champion-matchups-table"))
    )
    matchups_soup = BeautifulSoup(
        matchups_table.get_attribute("innerHTML"), "html.parser")
    rows = matchups_soup.find_all("div", {"class": "rt-tr-group"})

    winrate_dict = dict()

    example_patch = "10.10"  # scrap this later down the line

    # print(champ_name, role)

    for row in rows:
        features = row.find_all("div", {"class": "rt-td"})
        processed_champ = process_champ_name(features[1].text)
        winrate = float(features[2].text[:-1])
        gold_ft = int(features[3].text)
        xp_ft = int(features[4].text)
        kills_ft = float(features[5].text)
        cs_ft = float(features[6].text)
        jungle_cs_ft = float(features[7].text)
        num_matches = int("".join(features[8].text.split(",")))

        matchup_info = MatchupInfo(example_patch, winrate, gold_ft, xp_ft,
                                   kills_ft, cs_ft, jungle_cs_ft, num_matches)

        winrate_dict[(processed_champ, role)] = matchup_info

    return winrate_dict

# scrap synergies for a champion in a given role
# note: top & mid only have synergies for juglers
# bot only has synergies for supports & vice versa
# junglers have synergies with BOTH mid and top (not tracked currently)


def scrap_champ_synergies(champ_name, role):
    url = "https://u.gg/lol/champions/" + champ_name + "/duos?role=" + role
    browser.get(url)
    print("starting ", champ_name)
    print(url)
    duos_table = WebDriverWait(browser, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "champion-duos-table"))
    )
    duos_soup = BeautifulSoup(
        duos_table.get_attribute("innerHTML"), "html.parser")
    rows = duos_soup.find_all("div", {"class": "rt-tr-group"})

    synergy_dict = dict()

    example_patch = "10.10"  # scrap this later down the line

    for row in rows:
        features = row.find_all("div", {"class": "rt-td"})
        synergy_role = features[1].find_all("img")[0]["alt"]
        synergy_champ = features[2].text
        processed_champ = process_champ_name(synergy_champ)
        winrate = float(features[3].text[:-1])
        gold_ft = 0
        if features[4].text != "":
            gold_ft = int("".join(features[4].text.split(",")))
        xp_ft = 0
        if features[5].text != "":
            xp_ft = int("".join(features[5].text.split(",")))
        kills_ft = float(features[6].text)
        cs_ft = float(features[7].text)
        team_gold_ft = int("".join(features[8].text.split(",")))
        carry_perc = int(features[9].text[:-1])
        duo_carry_perc = int(features[10].text[:-1])
        num_matches = int("".join(features[8].text.split(",")))

        duo_info = DuoInfo(example_patch, winrate, gold_ft, xp_ft, kills_ft,
                           cs_ft, team_gold_ft, carry_perc, duo_carry_perc,
                           num_matches)

        synergy_dict[(processed_champ, synergy_role)] = duo_info

    return synergy_dict

# master scrapers per role


def scrap_all_synergies_per_role():
    roles = ["top", "jungle", "middle", "adc", "support"]
    master_synergies_dict = dict()

    for role in roles:
        print("scraping role", role)
        champs = scrap_role_champs(role)
        print("len: ", len(champs))
        for champ in champs:
            synergy_dict = scrap_champ_synergies(champ, role)
            master_synergies_dict[(champ, role)] = synergy_dict
        print()

    export_pickle(master_synergies_dict, "master_synergies_dict.pickle")


def scrap_all_matchups_per_role():
    roles = ["top", "jungle", "middle", "adc", "support"]
    master_matchups_dict = dict()

    for role in roles:
        print("scraping role", role)
        champs = scrap_role_champs(role)
        print("len: ", len(champs))
        for champ in champs:
            winrate_dict = scrap_champ_matchups(champ, role)
            master_matchups_dict[(champ, role)] = winrate_dict
        print()

    export_pickle(master_matchups_dict, "master_matchups_dict.pickle")


# functions to make dictionary symmetric

def make_matchups_symmetric(matchups_dict):
    for champ_tuple in list(matchups_dict.keys()):
        value = matchups_dict[champ_tuple]
        for champ_tuple_2 in value.keys():
            if champ_tuple_2 not in matchups_dict or \
                    champ_tuple not in matchups_dict[champ_tuple_2]:
                og_matchup = matchups_dict[champ_tuple][champ_tuple_2]
                new_matchup_info = \
                    MatchupInfo(
                        patch=og_matchup.patch,
                        winrate=100-og_matchup.winrate,
                        gold_ft=-og_matchup.gold_ft,
                        xp_ft=-og_matchup.xp_ft,
                        kills_ft=-og_matchup.kills_ft,
                        cs_ft=-og_matchup.cs_ft,
                        jungle_cs_ft=-og_matchup.jungle_cs_ft,
                        num_matches=og_matchup.num_matches
                    )
                if champ_tuple_2 not in matchups_dict:
                    matchups_dict[champ_tuple_2] = dict()
                matchups_dict[champ_tuple_2][champ_tuple] = new_matchup_info

    print("matchups made SYMMETRIC")


def make_synergies_symmetric(master_synergy_dict):
    for champ_tuple in list(master_synergy_dict.keys()):
        value = master_synergy_dict[champ_tuple]
        for champ_tuple_2 in value.keys():
            if champ_tuple_2 not in master_synergy_dict or \
                    champ_tuple not in master_synergy_dict[champ_tuple_2]:
                og_duo = master_synergy_dict[champ_tuple][champ_tuple_2]
                new_duo = \
                    DuoInfo(
                        patch=og_duo.patch,
                        winrate=og_duo.winrate,
                        gold_ft=og_duo.gold_ft,
                        xp_ft=og_duo.xp_ft,
                        kills_ft=0,  # this one is off
                        cs_ft=og_duo.cs_ft,
                        team_gold_ft=og_duo.team_gold_ft,
                        carry_perc=og_duo.duo_carry_perc,
                        duo_carry_perc=og_duo.carry_perc,
                        num_matches=og_duo.num_matches
                    )
                if champ_tuple_2 not in master_synergy_dict:
                    master_synergy_dict[champ_tuple_2] = dict()
                master_synergy_dict[champ_tuple_2][champ_tuple] = new_duo

# functions to make check dictionary is symmetric and error is smol


def check_matchups_symmetric(master_matchups_dict):
    for champ_tuple in list(master_matchups_dict.keys()):
        value = master_matchups_dict[champ_tuple]
        # print(champ_tuple, value)
        for champ_tuple_2 in value.keys():
            if champ_tuple_2 not in master_matchups_dict or \
                    champ_tuple not in master_matchups_dict[champ_tuple_2]:
                #print("NOT SYMMETRIC: ", champ_tuple, champ_tuple_2)
                pass
            else:
                wr1 = master_matchups_dict[champ_tuple][champ_tuple_2].winrate
                wr2 = master_matchups_dict[champ_tuple_2][champ_tuple].winrate
                if 100 - wr1 - wr2 > 1:
                    print("SYMMETRIC, sum ", wr1 + wr2)


def check_synergies_symmetric(master_synergy_dict):
    for champ_tuple in master_synergy_dict:
        value = master_synergy_dict[champ_tuple]
        for champ_tuple_2 in value.keys():
            if champ_tuple_2 not in master_synergy_dict or \
                    champ_tuple not in master_synergy_dict[champ_tuple_2]:
                # print("NOT SYMMETRIC", champ_tuple, champ_tuple_2)
                pass
            else:
                sn_1 = master_synergy_dict[champ_tuple][champ_tuple_2].winrate
                sn_2 = master_synergy_dict[champ_tuple_2][champ_tuple].winrate
                if abs(sn_1 - sn_2) > 1:
                    print("SYMMETRIC", sn_1, sn_2)

if __name__ == "__main__":
    scrap_all_matchups_per_role()
    master_matchups_dict = read_pickle("master_matchups_dict.pickle")
    make_matchups_symmetric(master_matchups_dict)
    check_matchups_symmetric(master_matchups_dict)
    export_pickle(master_matchups_dict, "master_matchups_dict.pickle")

    scrap_all_synergies_per_role()
    master_syngeries_dict = read_pickle("master_synergies_dict.pickle")
    make_synergies_symmetric(master_syngeries_dict)
    check_synergies_symmetric(master_syngeries_dict)
    export_pickle(master_matchups_dict, "master_synergies_dict.pickle")


browser.quit()
