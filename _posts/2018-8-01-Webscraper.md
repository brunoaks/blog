---
layout: post
title: Building a web scraper with Python and Beautiful Soup from scratch
---

The task of searching for the perfect grad school in the US is a daunting one. With so many different rankings and variables to consider for application, such as faculty size, tuition costs and location, one can easily spend a tremendous amount of time accessing each school's website and gathering information by hand.

I figured this task would be a perfect training ground to build my first web scraper. By using a centralized source of information about different schools - such as a universites' ranking website - I would be able to quickly scrape for data which would otherwise take ages to collect, if I were to copy and paste each bit of information into an Excel spreadsheet.

I decided to stick to the [USNews](https://www.usnews.com/best-graduate-schools/top-engineering-schools/eng-rankings) ranking for Graduate Schools in Engineering (because that's what fits me the most, if I ever go to grad school). As for tools, I used the [Requests](http://docs.python-requests.org/en/master/) library to get each page's HTML, and [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) for pulling data out of the HTML files. Then, I would write all the desired data into a .csv file.

Check out my [Github repo](https://github.com/brunoaks/the-masters-algorithm) for more details on the source code.

### 1. Creating the scraping strategy
Before setting off to code, I needed a web scraping strategy. This included visiting the USNews ranking and deciding the exact path my scraper would have to follow to obtain all the variables I wanted.

After analyzing the way the website is structured, I realized there was very limited info on the ranking page itself, so I would also need access each school's page inside USNews in order to get all the data I wanted.

This is the structure of the ranking page:

![scrape1](https://raw.githubusercontent.com/brunoaks/blog/master/images/scrape1.JPG)

And this is an overview of each engineering school's page:

![scrape2](https://raw.githubusercontent.com/brunoaks/blog/master/images/scrape2.JPG)

Since you can access each school's page from the original ranking, it would be a fairly easy task of just scraping the desired "href" tag in the ranking page's HTML and using it to generate another request for a new page. We would then iterate this process for every school we wanted.

I decided to limit my search to the Top 75 Engineering schools, and this meant requesting pages 1, 2 and 3 from the USNews ranking.

Thus, the general structure of the we scraper, represented in pseudocode, would look like this:

```
Import necessary packages
Initialize the .csv writer
Initialize lists for data storage
Create list of URLs from the ranking's pages
For each URL in list_of_URLs:
    Request HTML
    Transform HTML into "soup" file
    Save all desired info into feature lists
    Save links to each school's page into list
For each link in list_of_links:
    Request HTML
    Transform HTML into "soup" file 
    Save each school's desired info into feature lists
For each element in feature list:
    Save element into a "record" list
    Write "record" list as a row in a .csv file
```


### 2. Code away!
The code itself is very straightforward. The Requests library is extremely simple to use, and Beautiful Soup also makes it really easy to parse HTML files once you get the hang of it. After a while, it only becomes a matter of identifying the information you want inside each HTML file and choosing the appropriate command following the Beautiful Soup library.

You can check out the full code in Python 3 for the script below.

```python
from urllib.request import Request
import requests
from bs4 import BeautifulSoup as soup
import re
import csv
import pandas as pd

# Initializing writer 
csv_file = open('database.csv', 'w', buffering=1, encoding='utf8')
csv_file.write('\ufeff')
writer = csv.writer(csv_file, lineterminator = '\n')

# List of URLs to be scraped
urls = ['https://www.usnews.com/best-graduate-schools/top-engineering-schools/eng-rankings',
        'https://www.usnews.com/best-graduate-schools/top-engineering-schools/eng-rankings/page+2',
        'https://www.usnews.com/best-graduate-schools/top-engineering-schools/eng-rankings/page+3']

# Initializing features to be collected (each list will be a column in the final .csv file, except "univ_pages")
names = []
locations = []
univ_pages = []
inter_deadlines = []
inter_fees = []
ft_teachers = []
ft_enrollments = []
ft_tuitions = []
admissions_links = []
records = []

# Setting agent for scraping authentication
headers = {'User-Agent':'Mozilla/5.0'}

# Loop over the list of URLs
for url in urls:
    
    # Requesting page and saving them as soup objects
    req = requests.get(url, headers=headers)
    data = soup(req.text, "html5lib")
    
    # Appending each found value to the first three features
    for name in data.findAll('a', attrs={'class': 'school-name'}):
        names.append(name.string)
    for location in data.findAll('p', attrs={'class': 'location'}):
        locations.append(location.string)
    for univ_page in data.findAll('a', attrs={'class': 'school-name', 'href': True}):
        univ_pages.append(univ_page['href'])
        
# Extracting information from new URLs, which specific to each university
for univ_page in univ_pages:    
    new_url = 'https://www.usnews.com' + univ_page
    req2 = requests.get(new_url, headers=headers)
    data2 = soup(req2.text, "html5lib")
    
    # Collecting international deadline data
    inter_deadline = data2.find('td', attrs={'class':"column-last", 'data-test-id':"v_international_deadline"})
    try:    
        inter_deadlines.append(inter_deadline.text.strip())
    except:
        inter_deadlines.append(None)
    
    # Collecting international application fee data
    inter_fee = data2.find('td', attrs={'class':"column-last", 'data-test-id':"intl_application_fee"})
    try:
        inter_fees.append(inter_fee.text.strip())
    except:
        inter_fees.append(None)
    
    # Collecting full-time faculty data
    ft_teacher = data2.find('td', attrs={'class':"column-last", 'data-test-id':"ft_teachers"})
    try:
        ft_teachers.append(ft_teacher.text.strip())
    except:
        ft_teachers.append(None)
        
    # Collecting full-time enrollment data
    ft_enrollment = data2.find('td', attrs={'class':"column-last", 'data-test-id':"v_ft_enrolled_dir_pg"})
    try:
        ft_enrollments.append(ft_enrollment.text.strip())
    except:
        ft_enrollments.append(None)
        
    # Collecting full-time tuition data
    ft_tuition = data2.find('tr', attrs={'class':"extra-row v_mas_tuition_ft-extra-row"})
    try:
        ft_tuitions.append(ft_tuition.text.strip())
    except:
        ft_tuitions.append(None)
        
    # Collecting admissions links
    admissions_link = data2.find('a', attrs={'class':"t-strong", 'id': 'moreinfo_link', 'href':True})['href']
    try:
        admissions_links.append(admissions_link)
    except:
        admissions_links.append(None)    

# Add titles
titles = ['Rank', 'School', 'Location', 'Application Deadline', 
          'Application Fee', 'Full-time Faculty', 'Full-time enrollment', 
          'Full-time tuition', 'Grad admissions link']

# Write table titles
writer.writerow(titles)

# Add data to be written in .csv file, row by row
for i in range(len(names)):
    records.extend([i+1,
                    names[i],
                    locations[i],
                    inter_deadlines[i],
                    inter_fees[i],
                    ft_teachers[i],
                    ft_enrollments[i],
                    ft_tuitions[i],
                    admissions_links[i]])
    
    # Writing data into .csv file and reinitilizing storage variable
    writer.writerow(records)
    records = []
    
# Reading as a pandas DataFrame    
df = pd.read_csv('database.csv')
df.head()
```

### 3. Final thoughts
Running the script overwrites the .csv file called `database` with updated information. Unfortunately, not all desired variables were obtained, since USNews doesn't provide the same information for all schools (for instance, the "Applications deadline" variable was probably the one with the most "N/A" entries). But overall, the scraper seems to work well!

I actually had more fun with this little web scraping project than I initially thought. As one friend of mine told me, it feels like a genuine little puzzle. You already have an idea of what you need to collect - but you have to find that data hidden in layers of `div` tags and think of the optimal way of extracting those bits of information.

I really suggest you try it someday. Think of something that might actually be useful for you - as this grad school database is for me - and try to automate the process of data collection. You might save yourself a lot of dumb, repetitive work and learn a lot in the process.