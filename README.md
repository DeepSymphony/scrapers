# What this project is
It contains all logic for scraping and preprocessing our music data.

# Getting started
We use `virtualenv` to keep our dependencies clean. We use `python3` and `pip3`
to install and run this project.

```py
sudo pip install virtualenv # If you didn't install it
virtualenv -p python3 /your/path/to/the/virtual/env
source  /your/path/to/the/virtual/env/bin/activate
pip3 install -r requirements.txt  # Install dependencies
# Work on the project
# If you install something new and want to update requirements.txt
pip3 freeze > requirements.txt
deactivate # Exit the virtual environment
```

# Scraper
We currently scrape the **Midkar** dataset. To call the scraper, run
`python3 midkar_scraper.py`. It sleeps for 0.5 seconds every request because we
are polite scrapers!

# Clustering / Track Segmentation
Now that we scraped the dataset, it's time to find "beats" or "phrases" of
music within a MIDI track. We use a sliding window algorithm to break a track
up into multiple tracks. Run `python3 get_clusters.py` to cluster a 
**single** MIDI track **given a melody track id**. This needs to be fixed in the
future to cluster multiple MIDI tracks with melody track id labels. Dumps all
pickled clusters into `preprocess/midkar/pickle` and isolated melody MIDI into
`preprocess/midkar/melody`.

# Cluster Analysis
After clustering, the script will read from the folder containing
pickled files and convert them all into MIDI files at `output/midkar`.
