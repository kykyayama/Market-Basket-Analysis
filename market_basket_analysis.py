#!/usr/bin/env python
# coding: utf-8

# ## KEY OBJECTIVE
# 
# #### ->Identify frequently co-occurring products in customer transactions.
# 
# #### ->Identify Central item for frequent associations.
# 
# #### ->Discover association rules that reveal meaningful connections between items.
# 
# #### ->Optimize product placement within the store based on association rules.

# In[1]:


pip install mlxtend


# ## LOADING LIBRARIES
# 

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


# ## wrangling the dataset

# In[3]:


def wrangle(file_path):
    df = pd.read_csv(file_path)
    #CONVERTING DATE TO DATE TIME USING PANDAS
    df['Date'] = pd.to_datetime(df['Date'])
    #CREATING A YEAR COLUMN FROM DATE DATE COLUMN 
    df['item_count'] = df.groupby('Date')['itemDescription'].transform('nunique')


   
    return df
                     


# In[4]:


df = wrangle('Market_Basket_Analysis_Groceries_dataset.csv')


# # EDA

# In[5]:


df.head().set_index('Date')


# In[6]:


tab_info=pd.DataFrame(df.dtypes).T.rename(index={0:'column type'})
tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'null values (nb)'}))
tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100).T.rename(index={0:'null values (%)'}))
tab_info


# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


plt.figure(figsize =(15,10))
sns.histplot(df['Member_number'], bins=30, kde=True, color ="blue")
plt.title("member_number_distribution")
plt.show()


# this chart show an uniform distribution of memeber_number column Since there is no significant dominant customer group, this analysis is will base on capture diverse purchasing behaviors across the entire customer base uncovering general patterns, the number of transactions and associations rather than focusing on specific customer segments. 

# In[11]:


# using  z-scores to check the frequency of customer purchase of a particular item that are below and above mean.
df['item_count_zscore'] = (df['item_count'] - df['item_count'].mean()) / df['item_count'].std()
df.head(10)


# In[12]:


plt.figure(figsize =(15,10))
sns.histplot(df['item_count_zscore'], bins=30, kde=True, color ="blue")
plt.title("distribution _of_Zscores")
plt.show()


# In[13]:


niche_item =df[df['item_count_zscore']<-2]
niche_item


# In[14]:


niche_item.value_counts()


# Among the 682 items analyzed, comprising 1.75% of the dataset, those with low z-scores indicate a notably lower frequency of purchase. These items are representative of niche products and uncommon combinations that are infrequently included in customer baskets.

# In[15]:


popular_staple =  df[df['item_count_zscore']>2]
popular_staple 


# In[16]:


popular_staple .value_counts()


# Out of the 990 items examined, constituting 2.6% of the item_count, item_count with high z-scores indicate a considerably higher frequency of purchase. These items serve as popular staples commonly found in many baskets.

# In[17]:


frequent_item =df['itemDescription'].value_counts()
frequent_item


# In[18]:


sol_20_item=frequent_item.head(20)


# In[19]:


# Plot the top most frequent items
plt.figure(figsize=(15, 8))
sns.barplot(x=sol_20_item.index, y=sol_20_item.values, palette='viridis')
plt.title('Top 20 Most Frequent Items')
plt.xlabel('Items')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[20]:


dff = df['itemDescription'] == ("whole milk")
dff.value_counts()


# In[21]:


dff = df['itemDescription'] == ("pork")
dff.value_counts()


# In the presented bar chart, the highest sales were observed for whole milk, reaching approximately 2502 units, constituting around 6.45% of the total sales. Following closely, other vegetables accounted for 1898 units, while roll/buns trailed with 1716 units. Notably, pork recorded the lowest sales among the top 20 items, totaling 566 units.

# In[22]:


sols_20_item =frequent_item.tail(20)


# In[23]:


#Plot the least most frequent items
plt.figure(figsize=(15, 8))
sns.barplot(x=sols_20_item.index, y=sols_20_item.values, palette='viridis')
plt.title('least 20 Most Frequent Items')
plt.xlabel('Items')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[24]:


dff = df['itemDescription'] == ("preservative products")
dff.value_counts()


# In[25]:


dff = df['itemDescription'] == ("cooking chocolate")
dff.value_counts()  


# In the basket market analysis, no preservation products or kitchen utensil  were sold, 4 bags were sold, and cooking chocolate, with a mere 15 units sold, ranked highest as one of the purchased items among the bottom 20.

# In[26]:


# Creating unique transactions for each customer based on the items purchased by date
df['Transaction'] = df['Member_number'].astype(str) + '-' + df['Date'].astype(str)


# In[27]:


# Crosstab is used to create a frequency table of the transactions
df_all_item = pd.crosstab(df['Transaction'], df['itemDescription'])
df_all_item .head()


# In[28]:


# Encoding values to 0 and 1
    
df_basket = df_all_item .applymap(lambda x: 1 if x > 0 else 0)


# # Generating associate rule using apriori algorithm 

# 0.005 minimum support was used to enable the discover of items which z-scores fall outside the expected range (-2 to 2), inorder to consider a lower support threshold and discover associations involving those less frequent items.

# In[29]:


#Generate frequent itemsets with minimum support of 0.005 
frequent_itemsets = apriori(df_basket, min_support=0.005,use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift")
rules.sort_values(by = 'zhangs_metric', ascending = False)
rules.head(10)


# Milk as an antecedent has a strong associate with other items:
# (whole milk) -> (bottled beer) and (bottled beer) -> (whole milk): These rules have perfect confidence (0.999), indicating a very strong association between these items. They are also the most frequent rules.
# (whole milk) -> (canned beer) and (canned beer) -> (whole milk): While their confidence (0.811) is lower, the lift scores above 1 suggest a positive correlation.
# Other rules with whole milk: Whole milk appears in 17 of the top 20 rules, suggesting it's a central item with frequent associations.
# Notable Patterns:
# Meat and Vegetables: Rules like (other vegetables) -> (frankfurter) and (sausage) -> (other vegetables) suggest a common pairing of meat and vegetables.
# Breakfast Items: Rules like (whole milk) -> (domestic eggs) and (whole milk) -> (rolls/buns) point to potential breakfast combinations.

# In[30]:



# Scatter plot with confidence and lift as axes
plt.figure(figsize=(10, 6))
plt.scatter(rules['confidence'], rules['lift'], s=50, alpha=0.8)
plt.xlabel('Confidence')
plt.ylabel('Lift')
plt.title('Association Rule Scatter Plot')
plt.show()


# In the market basket analysis, the scattered plot reveals areas characterized by high lift and confidence levels signifying robust and reliable associations. Given a support threshold of 0.005, this association occurs in approximately 0.5% of transactions, indicating a moderate level of co-occurrence. Items situated in the upper right quadrant warrant prioritization, as they exhibit a complementary relationship.
# 

# In[31]:


# Heatmap to visualize rule properties
rules_pivot = rules.pivot(index='antecedents', columns='consequents', values='lift')
plt.figure(figsize=(12, 8))
sns.heatmap(rules_pivot, annot=True, cmap='coolwarm')
plt.title('Association Rule Heatmap (Lift)')
plt.show()


# People who buy other vegetables are 11.6% more likely to also buy frankfurters than expected by chance, The rule holds true in both directions, with similar metrics. This positive correlation suggests a complementary relationship between these items.same rules hold to yogurt and sausage, with a lift of 1.1 is a key indicator of this positive association. sausage and soda has a lift of 1.0 which is also a positive association and a complementary relationship similar metrics goes for milk and bottled beer. 

# In[32]:


# Filter association rules for co-purchasing opportunities
rules_4_sales = rules[(rules['antecedents'].apply(len) == 1) & (rules['consequents'].apply(len) == 1)]

# Sort rules based on confidence and support
rules_4_sales = rules_4_sales.sort_values(by=['confidence', 'support'], ascending=False)
# Select top cross-sale recommendations
top_ass_sale = rules_4_sales.head(10)

# Display cross-sale recommendations
print("Central item for frequent associations.:")
for idx, row in top_ass_sale.iterrows():
    antecedent = list(row['antecedents'])[0]
    consequent = list(row['consequents'])[0]
    print(f"Customers who purchased  '{antecedent}' also purchased '{consequent}'.")


# Several central items exhibit frequent associations with 'whole milk,' indicating significant co-purchasing patterns among customers. These associations highlight key items that are frequently bought together with 'whole milk,' providing valuable insights for marketing strategies, product placements, and customer preferences in the retail environment.

# In[33]:


#creating a Zhangs_metric for more Intentional purchase 
rules_zhang= rules[rules['zhangs_metric'] > 0]


# In[34]:


# Creating matrix and heatmap of for item with positive association and complementary relationship
association = rules_zhang.pivot(index='antecedents', columns='consequents', values='zhangs_metric').fillna(0)
plt.figure(figsize=(10, 10))
sns.heatmap(association, annot=True, cmap='coolwarm')
plt.title('Association Rules Heatmap: Frequent Itemset')
plt.show()


# Pairs such as 'other vegetable' and 'frankfurter,' 'sausage' and 'soda,' and 'sausage' and 'yogurt' emerge as swiftly adopted combinations in the market. These pairings exhibit a reliable association and a robust complementary relationship, suggesting a strong inclination among customers to purchase these items together. This insight can guide strategic decisions for product placement and promotions to enhance customer satisfaction and drive sales

# In[35]:


# Scatter plot of support vs. confidence for association rules
plt.figure(figsize=(10, 6))
plt.scatter(rules['support'], rules['confidence'], alpha=0.5, s=30)
plt.title('Support vs. Confidence for Association Rules')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.show()


#  ## experimenting with different support mining threshold

# In[36]:


#This filters rules with confidence less than 0.10 and support greater than 0.014.
high_conf_high_supp_rules = rules[(rules['confidence'] < 0.50) & (rules['support'] < 0.014)]
plt.scatter(high_conf_high_supp_rules['support'], high_conf_high_supp_rules['confidence'], alpha=0.5, s=30)
plt.title('Higher Support vs. higher Confidence for Association Rules')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.show()


#This filters rules with confidence less than 0.90 and support equal to 0.005 which is been used.
support_threshold = 0.005
high_conf_high_supp_rules = rules[(rules['confidence'] < 0.90) & (rules['support'] > support_threshold)]
plt.scatter(high_conf_high_supp_rules['support'], high_conf_high_supp_rules['confidence'], alpha=0.5, s=30)
plt.title('Higher Support vs. higher Confidence for Association Rules')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.show()


#This filters rules with confidence less than 0.90 and support greater than 0.014.
high_conf_high_supp_rules = rules[(rules['confidence'] < 0.90) & (rules['support'] > 0.014)]
plt.scatter(high_conf_high_supp_rules['support'], high_conf_high_supp_rules['confidence'], alpha=0.5, s=30)
plt.title('Higher Support vs. higher Confidence for Association Rules')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.show()


# Enforcing stringent rules in market basket analysis leads to increased specificity and reliability in the identified patterns. However, for our dataset when a higher support threshold exceeds 0.013% with a higher confidence, the rules become excessively strict, resulting in the absence of specific patterns and no discernible data displayed on the chart. This discrepancy prompted the adoption of a 0.005 support threshold for mining, as it yields more meaningful and actionable rules. The set of plots displays data points, indicating the presence of rules that meet the defined confidence and support criteria in the market basket analysis.

# In[37]:


df_support = rules[rules['support']> 0.013]
df_support


# In[38]:


#This filters rules with confidence less than 0.90 and support greater than 0.013.
high_conf_high_supp_rules = rules[(rules['confidence'] < 0.90) & (rules['support'] >0.013)]
plt.scatter(high_conf_high_supp_rules['support'], high_conf_high_supp_rules['confidence'], alpha=0.5, s=30)
plt.title('Higher Support vs. higher Confidence for Association Rules')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.show()


# Whole milk rolls/buns are the only two item that adhere to the most strict rule with support mining threshold up greater than  0.013 in market basket. If someone buys "whole milk," there's a 1.3968% chance they will also buy "rolls/buns." The confidence is 8.8447%, and the lift is 80.402. If someone buys "rolls/buns," there's a 1.3968% chance they will also buy "whole milk." The confidence is 12.6974%, and the lift is 80.4028. The negative values in the leverage, conviction, and zhangs_metric columns indicate that the observed co-occurrence is lower than expected if the items were independent.

# ## Recommendations

# ### Focus on broad customer base and popular items:
# 
# Leverage the uniform customer distribution: 
# Develop marketing campaigns and promotions targeting the entire customer base instead of segmenting them. Highlight "popular items".
# Personalize the shopping experience: Use customer purchase data to personalize product recommendations and promotions. Offer relevant coupons or discounts based on individual preferences.
# Gather customer feedback: Conduct surveys or focus groups to understand customer motivations and preferences behind observed associations. This can help refine marketing strategies and product offerings.
# 
# Capitalize on high-selling items: Emphasize top-selling items like whole milk, vegetables, rolls/buns in promotions and Create eye-catching displays. Consider bundled offers of top selling items + niche items.
# 
# 1. Healthy Breakfast Bundle:
# 
# Organic milk (niche)
# Whole wheat rolls/buns (top-selling)
# Fresh berries (niche)
# Shopping bags (top-selling)
# Yogurt (top-selling)
# Honey (niche)
# 2. Family Comfort Bundle:
# 
# Frozen chicken (niche)
# Shopping bags (top-selling)
# Frozen vegetables (top-selling)
# Canned soup (niche)
# Baking soda (niche)
# kitchen utensils(niche)
# 3. Grilling Party Bundle:
# 
# Sausage (top-selling)
# Bottled beer (top-selling)
# Shopping bags (top-selling)
# Hot dog buns (top-selling)
# cooking chocolate (niche)
# Salad dressing (top-selling)
# 4. Cleaning Essentials Bundle:
# 
# Toilet cleaner (niche)
# Decalcifier(niche)
# Rubbing alcohol(niche)
# Shopping bags (top-selling)
# Paper towels (niche)
# 5. Pampering Package:
# 
# Baby wipes/lotion (niche)
# Makeup remover (niche)
# Shopping bags (top-selling)
# Bath bombs (niche)
# Liqueur (niche)
# creams(niche)
# 
# 
# ### Promote complementary pairings:
# 
# Leverage on Milk Associations:
# Leverage the strong association of whole milk with other items like bottled beer and canned beer. Create joint promotions or discounts to encourage customers to purchase these complementary products together.
# 
# Highlight associations in marketing and store layout: it is highly noteable that the topselling items are mostly in the food,and drink category.promote potential combinations like breakfast combination like Meat and Vegetables: other vegetables -> frankfurter and sausage -> other vegetables suggest.
# Breakfast Items: whole milk -> domestic eggs and whole milk -> rolls/buns" in promotions, ads, and product canter placement.
# Cross-sell and upsell: Recommend complementary items at checkout based on the customer's basket. Offer discounts, joint promotions or bundled offers to enhance customer satisfaction and increase sales bundles for buying paired items together.
# ### Utilize data insights strategically:
# 
# Optimize product placement: Arrange "popular items" and frequently paired items strategically in the store to encourage impulse purchases.
# Personalize customer experiences: Use purchase history data to personalize recommendations and offers, encouraging repeat purchases and basket size growth.
# Monitor and refine strategies: Track the effectiveness of your actions based on sales data and customer feedback. Continuously adapt your strategies for optimal results.
# ### Additional Recommendations:
# 
# Monitor and evaluate results: Track the effectiveness of implemented strategies based on sales data and customer feedback. Adjust your approach as needed to optimize results.
# Utilize advanced data mining techniques: Consider exploring more sophisticated techniques like clustering or segmentation to uncover deeper insights into customer behavior and purchasing patterns.
# Address niche items:While not dominant, niche items and uncommon combinations still exist. Consider targeted promotions or online marketplaces to cater to these specific customer segments.
# Investigate basket analysis outliers: Explore the reasons behind items like reservation products, kitchen utensil, cooking chocolate having low etc sales despite being in the bottom 20 purchased items. This could reveal potential market shifts or untapped opportunities.
# Analyze seasonality and trends: use time decompose analysis, consider potential seasonal variations in purchasing patterns and adjust strategies accordingly. Stay informed about industry trends to identify new opportunities.

# In[ ]:





# In[ ]:





# In[ ]:




