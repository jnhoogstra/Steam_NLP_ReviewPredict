# Classifying Reviews and Recommending Games on Steam 

**Authors:** Elliott Iturbe,  Jacob Hoogstra, Griffin Riner

## Overview
The goal of this project is to discover how to make the most profitable movie possible in order to make business recommendations to Microsoft. Descriptive analyses of a variety of data on previously released movies reveal that the most profitable movie Microsoft could make is an adventure movie with an NR rating, directed by Jean Negulsesco at Buena Vistas studios, with a runtime between 100 minutes and 131 minutes.

## Business Problem
Microsoft wants to open a movie studio that can compete with other successful movie studios. However, they lack knowledge about filmmaking. To help them in this area, we will find out how a variety of factors affect the popularity of movies and then offer recommendations.

## Data
We examined data on Steam Reviews for video games, data included steamid, appid, app_title, app_tags, review, fps, voted_up, clean_review. Depending on our diffrent phase of the project diffrent features where selected for our recommendation system: add title and app tags were used. But for our NLP models all feature.

## Methods
Our process started with organizing our data by combining and minipulating dataframes in order to creating a new one usable one. Data for our Nlp was then vectorized and then traintest split and ranthroughmuliple models until eventually we came to a conclusion for _____ to be used. As for recommendation system sorting by specific values, and merging dataframes. While modeling our data, we used descriptive statistics to create helpful visuals that displayed our findings. Overall, our descriptive analysis is absolutely essential for anyone who wants to succeed in the movie industry.

## Results

This visualization shows which genre of movie Microsoft should make based on which genre has the highest worldwide gross.

![graph1](./images/grouped_barplot_Seaborn_barplot_Python_corrected.png)

This is a visual of how Rotten Tomatoes user-generated ratings vary according to different MPAA ratings of adventure movies.

![graph2](./images/Rotten_tomatose_Ratings.png)

This graph shows the most profitable directors for adventure movies. We conclude that Jean Negulesco is by far the best choice.

![graph3](./images/Directors_and_Profit_for_Adventure_Movies.png)

Buena Vistas Studios ("BV") is responsible 65% of the top 20 grossing movies. If Microsoft is interested in using another studio to make their film, they should contact Buena Vistas.

![graph4](./images/top20_barplot_Seaborn_barplot_Python.png)

This is a comparison of runtime to revenue that reveals there is no true monetary value for creating a movie with a runtime outside of the shaded area.

![graph5](./images/Runtime_Comparison_line_added.png)

This graph shows the average rating of movies according to month of release. Because there is no significant difference between each month, we conclude release month doesn't really matter.

![graph6](./images/Month_and_Rating.png)

## Conclusions
We recommend that Microsoft uses Buena Vistas studios or models their own studio after BV practices and creates an adventure movie with an NR rating. They should also hire Jean Negulseso. The run time of the movie should be between 100 minutes and 131 minutes. Microsoft should not put time and money into securing any particular release month.

## Next Steps
This project did not explore all possible avenues of increasing movie profitability. In the future, analyses could be performed regarding the actors who Microsoft should hire, what kind of storylines are received best by audiences, and what aspects of a movie trailer cause people to develop interest in seeing a movie. Additionally, it could benefit Microsoft to learn about the state of the film industry after COVID-19, which shut down theatres for a long period of time and may have impacted audience interest in ways that are not known yet.
use data of own creation api/wepscraping no duplicates
use aws

## For More Information
Please review our full analysis in [our Jupyter Notebook](./Final/Notebook.ipynb) or our [presentation](./microsoftmovieanalysispowerpoint.pdf).

For any additional questions, please contact **Elliott Iturbe at eaiturbe@bsc.edu, Jacon Hoogstra at jnhoogstra@crimson.ua.edu, or Griffin Riner at gnriner@bsc.edu**

## Repository Structure

```
├── data                                  <- data files used for analyses
├── images                                <- visualizations created
├── Final Notebook.ipynb                  <- code written for project with explanation
├── microsoftmovieanalysispowerpoint.pdf  <- PDF version of powerpoint
└── README.md                             <- overview of project
```

