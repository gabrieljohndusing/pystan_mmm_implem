# pystan_mmm_implem

## Introduction

### Marketing Mix Models

Read my [Medium article](https://gabrieljdusing.medium.com/productionizing-a-marketing-mix-model-ea12980af939) about this project.

Marketing mix models (MMM) are used by advertisers to understand how their advertising spending affects a certain KPI, for example, sales or revenue.
This allows them to optimize their future media budget more effectively.
To this end, Return on Ad Spend (ROAS) and marginal Return on Ad Spend (mROAS) are the most important metrics.
If a certain media channel, say TV advertising, has a high ROAS, then spending more on this channel results in higher sales. In contrast, mROAS measures incremental sales as a result of an incremental ad spend in this media channel.
The effect of spending on advertising is not immediately apparent - there is a lag between allocating the funds and seeing an increase in sales. Another effect is that spending has diminishing returns, that is there is a point where increasing spending on advertising results in little to no effect on sales. This means that linear regression will not capture these effects well. This paper makes use of Bayesian methods with flexible functional forms (the Hill function from pharmacology) to account for the lag and shape effects of ad spend on sales.

An implementation of this model using PyStan can be found here, together with an application to a more complicated dataset incorporating 13 media channels and 46 control variables.

## Project Goals

The goal of the project is to help productionize this model by automating the following:

- Creation of dataframe from user supplied weekly advertising spending and sales data
- Append economic data from Statistics Canada
- Add indicators for whether there is a holiday in a given week
- Interest from the number of searches on Google regarding a given keyword
- Interactive visualizations