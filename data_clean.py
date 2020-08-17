#import libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import glob

products = []  # List to store name of the product

#request send to url
r = requests.get(
    "https://www.flipkart.com/search?q=tshirts+men+white&as=on&as-show=on&otracker=AS_Query_OrganicAutoSuggest_6_11_na_na_ps&otracker1=AS_Query_OrganicAutoSuggest_6_11_na_na_ps&as-pos=6&as-type=RECENT&suggestionId=tshirts+men+white&requestId=af464515-5541-4f41-933c-fd7bd1295bb0&as-searchtext=tshirts+men&page=3")

content = r.content

#Beautiiful soup helps extract the data
soup = BeautifulSoup(content)
for a in soup.findAll('a', href=True, attrs={'class': '_2mylT6'}):
    products.append(a.text)

data = pd.DataFrame({'Product Name': products})


#combine all files in to one file
os.chdir(r"E:\flipkart data\flipkart clean data")

extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])

#export to csv
combined_csv.to_csv("flipkart_clean_data.csv",encoding='utf-8')



flipkart_combined_data=pd.read_csv(r"E:\flipkart data\flipkart clean data\flipkart_clean_data",encoding='utf-8')

flipkart_combined_data.head()