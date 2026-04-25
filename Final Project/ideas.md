
Public Dataset from Criteo Research

https://ailab.criteo.com/ressources/
https://ailab.criteo.com/criteo-sponsored-search-conversion-log-dataset/



Criteo Sponsored Search Conversion Log Dataset


This dataset contains logs obtained from Criteo Predictive Search (CPS). CPS, offers an automated end-to-end solution using sophisticated machine learning techniques to improve Google Shopping experience using robust, predictive optimization across every aspect of the advertiser’s campaign. CPS in general has two main aims : (1) Retarget high-value users via behavioral targeting such that the bids are based on each user’s likelihood to make a purchase. (2) Increase ROI using a bidding strategy which incorporates the effects of product characteristics, user intent, device and user behavior.

Each row in the dataset represents an action (i.e. click) performed by the user on a product related advertisement. The product advertisement was shown to the user, post the user expressing an intent via an online search engine.  Each row in the dataset, contains information about the product characteristics (age, brand, gender, price), time of the click ( subject to uniform shift), user characteristics and device information. The logs also contain information on whether the clicks eventually led to a conversion (product was bought) within a 30 day window and the time between click and the conversion.

We believe that this dataset provides a benchmark to analyze new applications in the world of conversion modeling for online search advertising.

Content of this dataset
This dataset includes following files:

README.md
CriteoSearchData: the dataset itself (6.4 GB)
Data description
Header Information:
<Sale>,<SalesAmountInEuro>,<time_delay_for_conversion>,<click_timestamp>,<nb_clicks_1week>,<product_price> ,<product_age_group> ,<device_type>,<audience_id> ,<product_gender> ,<product_brand> ,<product_category(1-7)> ,<product_country>, <product_id> ,<product_title> ,<partner_id> ,<user_id>

This dataset represents a sample of 90 days of Criteo live traffic data. Each line corresponds to one click (product related advertisement) that was displayed to a user. For each advertisement, we have detailed information about the product. Further, we also provide information on whether the click led to a conversion, amount of conversion and the time between the click and the conversion.  Data has been sub-sampled and anonymized so as not to disclose proprietary elements.

Delimited: \t (tab separated)

Missing Value Indicator: -1 ( Missing value indicator is 0 for click_timestamp)

Outcome/Labels
Sale : Indicates 1 if conversion occurred and 0 if not).
SalesAmountInEuro : Indicates the revenue obtained when a conversion took place. This might be different from product-price, due to attribution issues. It is -1, when no conversion took place.
Time_delay_for_conversion : This indicates the time between click and conversion. It is -1, when no conversion took place.
Features
click_timestamp: Timestamp of the click. The dataset is sorted according to timestamp.
nb_clicks_1week: Number of clicks the product related advertisement has received in the last 1 week.
product_price: Price of the product shown in the advertisement.
product_age_group: The intended user age group of the user, the product is made for.
device_type: This indicates whether it is a returning user or a new user on mobile, tablet or desktop. 
audience_id:  We do not disclose the meaning of this feature.
product_gender: The intended gender of the user, the product is made for.
product_brand: Categorical feature about the brand of the product.
product_category(1-7): Categorical features associated to the product. We do not disclose the meaning of these features.
product_country: Country in which the product is sold.
product_id: Unique identifier associated with every product.
product_title: Hashed title of the product.
partner_id: Unique identifier associated with the seller of the product.
user_id: Unique identifier associated with every user.
Note :- All categorical features have been hashed.