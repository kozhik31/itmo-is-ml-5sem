import glob
import re
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd

DATA = "data.csv"
RE_PRICE = re.compile(r"\$(\d+.\d+)")
RE_RAM = re.compile(r"(\d+)\s*([A-Za-z]+)")
RE_HDD = re.compile(r"(\d+|\d+.\d+)\s*([A-Za-z]+)")
RE_RAM_2 = re.compile(r"(\d+)\s*([A-Za-z]+)\s*(ram|RAM)")
RE_HDD_2 = re.compile(r"(\d+|\d+.\d+)\s*([A-Za-z]+)\s*(Hard Drive|HD)")
RE_POSITIVE_REVIEWS = re.compile(r"(\d+)%")

main_genres = [
    "Action",
    "Adventure",
    "RPG",
    "Strategy",
    "Simulation",
    "Casual",
    "Puzzle",
    "Sports",
    "Racing",
    "Horror"
]


def parse_date(soup):
    div_date = soup.find("div", class_="date")
    if div_date:
        date = datetime.strptime(div_date.text, "%b %d, %Y")
        return date.timestamp()
    return -1


def parse_price(soup):
    div_price = soup.find("div", class_="game_purchase_price price")
    if div_price:
        if "Free To Play" in div_price.text:
            return 0.0

        price = RE_PRICE.search(div_price.text)
        return float(price.group(1))

    div_price = soup.find("div", class_="discount_original_price")
    if div_price:
        price = RE_PRICE.search(div_price.text)
        return float(price.group(1))

    return -1


def _parse_game_area_sys_req_leftCol(div_minimal):
    ram, hdd, measure_ram, measure_hdd = 0, 0, 0, 0
    ul = div_minimal.find_next("ul", class_="bb_ul")
    items = ul.find_all("li")

    for li in items:
        strong = li.find("strong")
        if strong and "Memory:" in strong:
            reg = RE_RAM.search(li.text)
            ram = int(reg.group(1))
            measure_ram = reg.group(2)
        elif strong and ("Hard Drive:" in strong or "Storage:" in strong):
            reg = RE_HDD.search(li.text)
            hdd = float(reg.group(1))
            measure_hdd = reg.group(2)

    return ram, measure_ram, hdd, measure_hdd


def _parse_game_area_sys_req_full_strong(items):
    ram, hdd, measure_ram, measure_hdd = 0, 0, 0, 0
    for p in items:
        strong = p.find("strong")
        if strong and "Minimum:" in strong:
            reg_ram = RE_RAM_2.search(p.text)
            reg_hdd = RE_HDD_2.search(p.text)
            if reg_ram:
                ram = int(reg_ram.group(1))
                measure_ram = reg_ram.group(2)
            if reg_hdd:
                hdd = float(reg_hdd.group(1))
                measure_hdd = reg_hdd.group(2)

    return ram, measure_ram, hdd, measure_hdd


def _parse_game_area_sys_req_full_ul(ul):
    ram, hdd, measure_ram, measure_hdd = 0, 0, 0, 0
    for li in ul.find_all("li"):
        if "Memory:" in li.text:
            reg_ram = RE_RAM.search(li.text)
            if reg_ram:
                ram = int(reg_ram.group(1))
                measure_ram = reg_ram.group(2)
        if "Hard Drive:" in li.text or "Storage:" in li.text:
            reg_hdd = RE_HDD.search(li.text)
            if reg_hdd:
                hdd = float(reg_hdd.group(1))
                measure_hdd = reg_hdd.group(2)

    return ram, measure_ram, hdd, measure_hdd


def _parse_game_area_sys_req_full_minimum(minimum):
    ram, hdd, measure_ram, measure_hdd = 0, 0, 0, 0
    ul = minimum.find_next("ul")
    if ul:
        for li in ul.find_all("li"):
            strong = li.find("strong")
            if strong and "Memory:" in strong:
                reg_ram = RE_RAM_2.search(li.text)
                if reg_ram:
                    ram = int(reg_ram.group(1))
                    measure_ram = reg_ram.group(2)
            if strong and ("Hard Drive:" in strong or "Storage:" in strong):
                reg_hdd = RE_HDD_2.search(li.text)
                if reg_hdd:
                    hdd = float(reg_hdd.group(1))
                    measure_hdd = reg_hdd.group(2)

    return ram, measure_ram, hdd, measure_hdd


def parse_sys_req(soup):
    div_minimal = soup.find("div", class_="game_area_sys_req_leftCol")
    ram, hdd, measure_ram, measure_hdd = 0, 0, 0, 0
    if div_minimal:
        ram, measure_ram, hdd, measure_hdd = _parse_game_area_sys_req_leftCol(div_minimal)
    else:
        div_minimal = soup.find("div", class_="game_area_sys_req_full")
        if div_minimal:
            ul = div_minimal.find_next("ul")
            items = ul.find_all("p")
            if items:
                ram, measure_ram, hdd, measure_hdd = _parse_game_area_sys_req_full_strong(items)
            elif ul := ul.find_next("ul"):
                ram, measure_ram, hdd, measure_hdd = _parse_game_area_sys_req_full_ul(ul)
            else:
                minimum = soup.find("strong", string="Minimum:")
                if minimum:
                    ram, measure_ram, hdd, measure_hdd = _parse_game_area_sys_req_full_minimum(minimum)

    if ram and (measure_ram == "GB" or measure_ram == "gb"):
        ram *= 1024
    if hdd and (measure_hdd == "GB" or measure_hdd == "gb"):
        hdd *= 1024

    return ram, hdd


def parse_reviews(soup):
    span_reviews = soup.find_all("span", class_="nonresponsive_hidden responsive_reviewdesc")
    if len(span_reviews) >= 2:
        review_30 = float(RE_POSITIVE_REVIEWS.search(span_reviews[0].text).group(1))
        review_all = float(RE_POSITIVE_REVIEWS.search(span_reviews[1].text).group(1))
        return review_30 / 100, review_all / 100
    elif len(span_reviews) == 1:
        review_all = float(RE_POSITIVE_REVIEWS.search(span_reviews[0].text).group(1))
        return 0.0, review_all / 100

    return 0.0, 0.0


def parse_genres(soup):
    genres = [0] * 10
    b_genre = soup.find("b", string="Genre:")
    if b_genre:
        span_genres = b_genre.find_next("span")
        if span_genres:
            a = span_genres.find_all("a")
            for genre in a:
                if genre.text in main_genres:
                    genres[main_genres.index(genre.text)] = 1

    return genres


def parse(path):
    f = open(path, encoding='utf-8')
    soup = BeautifulSoup(f, 'lxml')
    price = parse_price(soup)
    date = parse_date(soup)
    ram, hdd = parse_sys_req(soup)
    reviews_30, reviews_all = parse_reviews(soup)
    genres = parse_genres(soup)
    f.close()

    return [price, date, ram, hdd, reviews_30, reviews_all] + genres


columns = [
    "id", "price", "date", "ram", "hdd", "reviews_30", "reviews_all",
    "Action", "Adventure", "RPG", "Strategy", "Simulation",
    "Casual", "Puzzle", "Sports", "Racing", "Horror"
]


def main():
    data = []
    cnt = 1
    for path in glob.glob("templates/*"):
        try:
            data.append([cnt] + parse(path))
            cnt += 1
            if cnt % 50 == 0:
                print(cnt)
        except (AttributeError, ValueError) as ex:
            print(path)
            print(ex)

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(DATA, index=False, encoding="utf-8")


main()
