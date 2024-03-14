# CMI-sleep-state-detection

## Description
The goal of this project is to detect sleep onset and wake.<br>
Developed a ML model trained on wrist-worn accelerometer data in order to determine a person's sleep state.<br>
The work could make it possible for researchers to conduct more reliable, larger-scale sleep studies across a range of populations and contexts. The results of such studies could provide even more information about sleep.By accurately detecting periods of sleep and wakefulness from wrist-worn accelerometer data, researchers can gain a deeper understanding of sleep patterns and better understand disturbances in children.

## Context
The “Zzzs” you catch each night are crucial for your overall health. Sleep affects everything from your development to cognitive functioning. Even so, research into sleep has proved challenging, due to the lack of naturalistic data capture alongside accurate annotation. If data science could help researchers better analyze wrist-worn accelerometer data for sleep monitoring, sleep experts could more easily conduct large-scale studies of sleep, thus improving the understanding of sleep's importance and function.
<br>
Current approaches for annotating sleep data include sleep logs, which are the gold standard for detecting the onset of sleep. However, they are impractical for many participants to use reliably, and fail to capture the nuanced difference between heading to bed and falling asleep (or, conversely, waking up and getting out of bed). Heuristic-based software is another solution that attempts to identify sleep windows, though these rely on human-engineered features of sleep (i.e. arm angle) that vary across individuals and don’t accurately summarize the sleep windows that experts can visually detect from their data. With improved tools to analyze sleep data on a large scale, researchers can explore the relationship between sleep and mood/behavioral difficulties. This knowledge can lead to more targeted interventions and treatment strategies.
## Acknowledgments
The data used for this project was provided by the Healthy Brain Network, a landmark mental health study based in New York City that will help children around the world. In the Healthy Brain Network, families, community leaders, and supporters are partnering with the Child Mind Institute to unlock the secrets of the developing brain. In addition to generous support provided by the Kaggle team, financial support has been provided by the Stavros Niarchos Foundation (SNF) as part of its Global Health Initiative (GHI) through the SNF Global Center for Child and Adolescent Mental Health at the Child Mind Institute.

## Dataset Description
The dataset comprises about 500 multi-day recordings of wrist-worn accelerometer data annotated with two event types: onset, the beginning of sleep, and wakeup, the end of sleep. Your task is to detect the occurrence of these two events in the accelerometer series.
<br>
While sleep logbooks remain the gold-standard, when working with accelerometer data we refer to sleep as the longest single period of inactivity while the watch is being worn. For this data, we have guided raters with several concrete instructions:
<br>
A single sleep period must be at least 30 minutes in length.<br>
A single sleep period can be interrupted by bouts of activity that do not exceed 30 consecutive minutes<br>
No sleep windows can be detected unless the watch is deemed to be worn for the duration (elaborated upon, below)<br>
The longest sleep window during the night is the only one which is recorded<br>
If no valid sleep window is identifiable, neither an onset nor a wakeup event is recorded for that night.<br>
Sleep events do not need to straddle the day-line, and therefore there is no hard rule defining how many may occur within a given period. However, no more than one window should be assigned per night. For example, it is valid for an individual to have a sleep window from 01h00–06h00 and 19h00–23h30 in the same calendar day, though assigned to consecutive nights
There are roughly as many nights recorded for a series as there are 24-hour periods in that series.<br>
Though each series is a continuous recording, there may be periods in the series when the accelerometer device was removed. These period are determined as those where suspiciously little variation in the accelerometer signals occur over an extended period of time, which is unrealistic for typical human participants. Events are not annotated for these periods, and you should attempt to refrain from making event predictions during these periods: an event prediction will be scored as false positive.
<br>
Each data series represents this continuous (multi-day/event) recording for a unique experimental subject.
