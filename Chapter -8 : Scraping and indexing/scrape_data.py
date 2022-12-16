import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import argparse

# BASE_URL = "https://papers.nips.cc/paper/"

# result= requests.get(BASE_URL)
# soup = BeautifulSoup(result.content, "html.parser")
# print(soup.prettify())
# print()

BASE_URL = "https://papers.nips.cc/paper/"

def get_years_of_paper(url):
    try:
        response = requests.get(url)
        years = list()
        soup = BeautifulSoup(response.content,"html5lib")
        for li in soup.find('div',class_ = "col-sm").find_all("li"):
            years.append(li.a.get('href').split("/")[-1])
        print(f"Number of years in site : {len(years)}")
        
        return years
    except ConnectionError:
        print("Connect to Network")

def get_details(years):
    all_urls = list()
    all_titles = list()
    all_abstract = list()
    for year in tqdm(years,total=len(years)):
        site_url = f"https://papers.nips.cc/paper/{int(year)}"
        year_urls = get_year_urls(site_url)
        print("Extracting details year-wise")
        for url in tqdm(year_urls,total=len(year_urls)):
            response = requests.get(url)
            soup = BeautifulSoup(response.content,"html5lib")
            title = soup.find("div",class_ = "col").find_all("h4")[0].text
            abstract = soup.find("div",class_ = "col").find_all("p")[-2].text
            all_titles.append(title)
            all_abstract.append(abstract)
            all_urls.append(url)
    return all_urls,all_titles,all_abstract

def get_year_urls(url):
    SITE_URL = "https://papers.nips.cc"
    all_urls = list()
    response = requests.get(url)
    soup = BeautifulSoup(response.content,"html5lib")
    for li in soup.find("div",class_ = "col").find_all("li"):
        all_urls.append(SITE_URL+li.a.get('href'))
    print("Extracted URL's")
    return all_urls

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    args = parser.parse_args()
    years = args.year 
    print(years,args)

    years = get_years_of_paper(BASE_URL)
    all_urls,all_titles,all_abstract = get_details(years[:2])
    print(len(all_urls),len(all_titles),len(all_abstract))

    