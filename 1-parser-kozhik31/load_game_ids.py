import random
import re
from time import sleep

import requests
from bs4 import BeautifulSoup

RE_ID = re.compile(r"app/(\d+)/")

def get_ids():
    ids = set()
    for i in range(0, 400):
        url = f'https://store.steampowered.com/search/?ndl=1&page={i}'
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "lxml")

        for link in soup.find_all('a'):
            href = link.get('href')
            game = RE_ID.search(href)
            if game:
                ids.add(game.group(1))
        print(i)
        sleep(random.uniform(0.5, 1.0))

    return ids

ids = get_ids()

with open('ids.txt', 'a') as f:
    for id in ids:
        f.write(f"{id}\n")