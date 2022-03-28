Project | Project 5: Data visualization and reporting in Streamlit

Project Goal:

Creating a dashboard for a dataset of bank loans


Steps

- Step 1:potential problems of dataset

1.	Missing values

First, we looked if there a lot of missing values, we didn’t find any missing values in our dataset.

2.	Useless columns
 We choose to droppe the columns that we thought that will not be useful for our analysis, so we dropped the following columns:  
['Funded Amount Investor','Accounts Delinquent','Batch Enrolled', 'Payment Plan',  'Open Account', 'Public Record', 'Initial List Status', 'Collection 12 months Medical',  'Last week Pay', 'Delinquency - two years', 'Revolving Utilities', 'Total Received Interest',   'Recoveries', 'Total Received Late Fee', 'Collection Recovery Fee', 'Debit to Income',   'Sub Grade', 'Verification Status', 'Total Collection Amount', 'Inquires - six months']

3.	Outliers
we looked for the outliers and we found a lot of outlies. We decided to transform the outliers to NAN values and after that we dropped these values.

4.	Inconsistency 
We found a lot of inconsistency cases specially in Loan Title, therefor we created another new column called ‘Purpose_loan’ with regrouped categories more consistent with the analysis.

After these four steps we created a copy cleaned of our initial dataset suitable for further analysis.

Step 2: Dashboard development

The dashboard that we created, intitled ‘BANK LOANS DASHBOARD’, concerns an analysis of bank loans dataset. There are 67000 customers with 35 different columns related to characteristics of loans and borrowers. 
-Univariate Analysis
We started the analysis by a univariate analysis of different variables. 
-Loan amount
- Interest rate
- Home ownership
- Application type
-Loans grade

-	Bivariate analysis
We tried to analyze the correlation between different specific variables. we realized different crosstables according to loan amount and interest rate, then we calculated the correlation matrix to find the pairwise correlation of all columns in the dataframe.  




