from bs4 import BeautifulSoup
import requests
import urllib.request as ur
from time import sleep
from os.path import join
import os
# make a data dir to hold midi files
os.makedirs("./data/midkar", exist_ok=True)

base = "http://midkar.com/jazz/"
page_nums = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11",    "12", "13"]

for page_num in page_nums:
    url = "{}jazz_{}.html".format(base, page_num)
    print("Scraping page:", url)
    r = requests.get(url)
    data = r.text
    soup = BeautifulSoup(data, "html5lib")
    links = []

    # gather all file paths
    for link in soup.find_all('a'):
        filename = link.get('href')
        if filename and filename[-4:].lower() == ".mid":
            full_url = join(base, filename)
            links.append((filename, full_url))
    # download all file paths
    for name, link in links:
        print(" Downloading file:", link)
        retries = 5
        while retries > 0:
            try:
                sleep(0.5)
                if link[:4] == "http":
                    res = ur.urlopen(link)
                else:
                    res = ur.urlopen(base + link)
                with open("data/" + name, 'wb') as file:
                    file.write(res.read())
                break
            except Exception as e:
                retries -= 1
