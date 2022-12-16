import requests
import re 
from tqdm import tqdm
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm 
url = "https://nips.cc/Conferences/2022/Schedule?showEvent="

response = requests.get(url)
soup = BeautifulSoup(response.content,"html5lib")

all_show_details  = []
for data in soup.findAll("div",onclick=True):
    if "showDetail" in  data.get("onclick"):
        show_details = re.findall(r"\AshowDetail\([0-9]+\)",data.get("onclick"))
        all_show_details.extend(show_details)
print(len(all_show_details),len(set(all_show_details)))


event_ids = []
for show_details in all_show_details:
    res = re.findall("[0-9]+",show_details)
    event_ids.extend([int(x) for x in res])
print(len(event_ids),len(set(event_ids)))



def concurrency(function,data,length,method="thread"):
    if method=="thread":
        print("Mode :",method)
        with ThreadPoolExecutor(5) as pe:
           res = list(tqdm(pe.map(function,data),total=length))
           return res
    elif method=="process":
        print("Mode :",method)
        with ProcessPoolExecutor(5) as pe:
           res = list(tqdm(pe.map(function,data),total=length))
           return res
    else:
        print("ERROR")
        raise


base_event_url = "https://nips.cc/Conferences/2022/Schedule?showEvent="

def fetch_data_from_event_id(event_id):
    data = {"event_id":event_id}
    url_ = base_event_url+str(event_id)
    event_res = requests.get(url_)
    event_soup = BeautifulSoup(event_res.content,"html5lib")
    paper_title = event_soup.findAll("div",class_="maincardBody")
    abstract = event_soup.findAll("div",class_= "abstractContainer")
    data["title"] = paper_title[0].text
    data["abstract"] = abstract[0].text
    data["url"] = url_
    return data

results = concurrency(fetch_data_from_event_id,event_ids,len(event_ids))