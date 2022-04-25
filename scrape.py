import json
import time

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://bigbangtrans.wordpress.com/"

res = requests.get(BASE_URL)

soup = BeautifulSoup(res.content, features="html5lib")
links = [link.get("href") for link in soup.find_all("a")]
links = [link for link in links if "https://bigbangtrans.wordpress.com/series" in link]

print(links[:5])

data = []
for link in tqdm(links):
    res = requests.get(link, timeout=2)
    soup = BeautifulSoup(res.content, features="html5lib")
    title = soup.find("h2", attrs={"class": "title"}).text
    # entry = soup.find("div", attrs={"class": "entrytext"})
    scripts = soup.find_all("p")
    scripts = [script.text for script in scripts]
    scripts = [script.split(": ") for script in scripts if ":" in script]

    item = {
        "title": title,
        "scripts": scripts,
    }
    data.append(item)
    time.sleep(0.1)

json.dump(data, open("./data.json", "w"), indent=2)
