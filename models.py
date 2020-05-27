class DuoInfo:
    def __init__(self, patch, winrate, gold_ft, xp_ft, kills_ft, cs_ft, \
        team_gold_ft, carry_perc, duo_carry_perc, num_matches):
        self.patch = patch
        self.winrate = winrate
        self.gold_ft = gold_ft
        self.xp_ft = xp_ft
        self.kills_ft = kills_ft
        self.cs_ft = cs_ft
        self.team_gold_ft = team_gold_ft
        self.carry_perc = carry_perc
        self.duo_carry_perc = duo_carry_perc
        self.num_matches = num_matches

class MatchupInfo:
    def __init__(self, patch, winrate, gold_ft, xp_ft, kills_ft, cs_ft, \
        jungle_cs_ft, num_matches):
        self.patch = patch
        self.winrate = winrate
        self.gold_ft = gold_ft
        self.xp_ft = xp_ft
        self.kills_ft = kills_ft
        self.cs_ft = cs_ft
        self.jungle_cs_ft = jungle_cs_ft
        self.num_matches = num_matches