# MLB Injuries

A library for scraping MLB IL/DTD stints and modeling injury risk with Temporal Point Processes. The code in this repo produces the results and plots in [](https://sharpestats.com/mlb-injury-point-process/)

![df](https://sharpestats.com/assets/img/mlb-injury-point-process/trout-split-intensity.png)

---

### Installation

```
conda create -n "injury" python=3.8
pip install -e .
```


### Data Scraping

MLB IL data is available through the stats api and can be scraped using functions in `injury.scrape.statsapi`. Day to day information is available at Pro Sports through `injury.scrape.prosports`. 

Scraping and aligning of these two datasets is contained in `notebooks/scrape_align_injuries.ipynb`. 

### Injury Data Preprocessing

Injury mapping and event processing can be found in `injury/preprocess`. Injury data is cleaned from both sources, categorized, and combined with player game data to determine injury timing. 

### Modeling 

Injuries are modeled using a Hawkes Process, which inherently model a set of events through an intesity function:

![intensity](/img/hawkes.png))

The input to the model is the events and respective times of each event. For example:

```python 
{
    't': [642.0, 649.0, 932.0, 934.0, 953.0, 999.0, 1159.0, 1172.0],
    'injury_location': ['foot','wrist','hamstring','hamstring','hand','head/neck','wrist','other leg']
}
```

Models can be trained using the command 
```
python -m injury.model.train
```

There are available options for different types of models. For example, to train a model that splits IL and DTD events, on only pitchers, using a 70/30 train/test split use
```
python -m injury.model.train -p "pitcher" -t 0.3 --dtd
```

