select distinct player_id
, name
, game_date
, team 
FROM 
(select 
 distinct 
 batter as player_id
 , batter_name as name
 , game_date 
 , case when inning_topbot = 'Top'
  then away_team else home_team end as team
 from pbp where year between %(min_year)s and %(max_year)s
UNION
select  distinct 
 pitcher as player_id
 , pitcher_name as name
 , game_date 
 , case when inning_topbot = 'Bot' 
  then away_team else home_team end as team
from pbp where year between %(min_year)s and %(max_year)s
 ) as players
