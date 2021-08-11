# Classifying Reviews and Recommending Games on Steam 

**Authors:** Elliott Iturbe,  Jacob Hoogstra, Griffin Riner

## Overview
The goal of this project is to create an NLP model that can predict if a user would like or not like a game and then combine a recommendation system to the NLP model. In order to give recommendations to the client on games they would like based off the review they created for the previous game.

## Business Problem
We set out to make it easier for Users to determing what game to play based off of review they wrote for a previous game that our  NLP model runs and then passes to our recommendation system which recommends games the user would be intrested in.

## Data
We examined data on Steam Reviews for video games, data included steamid, appid, app_title, app_tags, review, fps, voted_up, clean_review. Depending on our diffrent phase of the project diffrent features where selected for our recommendation system: add title and app tags were used but for our NLP models all feature were used.

## Methods
Our process started with organizing our data by combining and minipulating dataframes in order to creating a new one usable one. Data for our NLP was then vectorized and then train test split and ran through muliple models until eventually we came to a conclusion for MultinomialNB model to be used. As for recommendation system sorting by specific values, and merging dataframes. While modeling our data, we used descriptive statistics to create helpful visuals that displayed our findings. Overall, our descriptive analysis is absolutely essential for anyone who wants to succeed in the movie industry.

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
This project used a premade data set that was flawed with duplicates and multiple games with not many reviews and one or two games with numerous reviews. So we would want to create our own dataset with the use of an Api call or wepscraping with no duplicates and ability to control to amount of reviews. As well as we would need to use AWS instead of working locally in order to use the large amounts of data we had instead of using samples of the data in order to work on the data locally. 


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

