## Processing Pipeline:
1. Download raw CSVs and store in **raw_data/**
2. Run **clean_data.py** to filter out the ignore=True rows and to make sure the track fixed column is the BOT ID
    - 'track_fixed' should be the bot's ID number (last number of bot ID)
    - 'ignore' should be False 
    - 'Run6group5subject5.csv', 'Run4group3subject2.csv', 'Run6group5subject1_2' had to be manually checked because the filename ID doesn't match the track_fixed ID
    - also ignores 'run8bot2_2.csv' (this csv contains many issues)
3. Cleaned data is stored in **cleaned_data/**
4. Run **trim_data.py** to trim flagged CSVs
    - cuts out parts of tracking videos at beginning or end that contain interference with other bots or flipping
5. Trimmed data is stored in **trimmed_data/**
6. Run **split_data/** to split each CSV into 30 second segments - store each segment as a new CSV
    - removed segments where there is a frame missing
7. Split CSVs are store in **split_data/BOT_ID_30sec_chunks/BOT_ID_run\*.csv**
8. Run **shift_rotate_segments.py** to rotate data so all segments begin with the same heading and shift data to start at (0,0)
    - make sure the most recent copy of **degrees_to_rotate_in_silico_bots.csv** is downloaded from google docs
    - saves out heading of bot during rotation to **in_vitro_data/headings.csv**

## Notes:
- Run **plot_full_trajectory.py** to plot full trajectories (raw, cleaned, trimmed)
- Run **plot_segmented_trajectory.py** to plot 30 second segments (raw, rotated/shifted)

 