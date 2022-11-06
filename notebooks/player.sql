with batters as (
  select 
 distinct 
 batter as player_id
 , batter_name as name
 , game_date 
 , game_type
 , case when inning_topbot = 'Top'
  then away_team else home_team end as team
 from pbp 
 where year between %(min_year)s and %(max_year)s
 and is_pa
),

pitchers as (
select  distinct 
 pitcher as player_id
 , pitcher_name as name
 , game_date
 , game_type 
 , case when inning_topbot = 'Bot' 
  then away_team else home_team end as team
from pbp 
where year between %(min_year)s and %(max_year)s
      and is_pa
),

pitcher_pas as (
select player_id
, count(*) as pa_pitcher
from pitchers
group by player_id
),

batter_pas as (
  select player_id
  , count(*) as pa_batter
  from batters
  group by player_id
)



select distinct players.player_id
, name
, game_date
, game_type
, team 
, case when pa_batter is null or pa_pitcher >= pa_batter 
  then 'pitcher' else 'batter' end as position
FROM 
(select * from batters UNION 
select * from pitchers)
as players
left join pitcher_pas pit
on players.player_id = pit.player_id
left join batter_pas bat
on players.player_id = bat.player_id
