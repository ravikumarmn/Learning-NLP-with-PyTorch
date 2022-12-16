import os 
from annoy import AnnoyIndex
import torch 
import json 
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pprint import pprint
import webbrowser
os.environ["TOKENIZERS_PARALLELISM"]= "false"

class AnnoyIndexer:
    def __init__(self,embeddings_path:str,data_path,model_str="all-MiniLM-L6-v2",trees=10) -> None:
        self.json_data = json.load(open(data_path,"r"))
        self.embeddings_dict = torch.load(embeddings_path)
        embed_size = self.embeddings_dict["embed_size"]
        embeddings = self.embeddings_dict["embeddings"]
        self.sbert = SentenceTransformer(model_str)
        ann_path   = embeddings_path.split("/")[:-1] + model_str.split("/")[-1:]
        ann_path   = "/".join(ann_path) + ".ann"
        path = Path(ann_path) 
        if path.is_file():
            self.ann = AnnoyIndex(embed_size, 'angular')
            self.ann.load(ann_path)
        else:
            self.ann = AnnoyIndex(embed_size, 'angular')
            for idx,embs in embeddings.items():
                self.ann.add_item(idx,embs)
            self.ann.build(trees)
            print("Saving ann: ",ann_path)
            self.ann.save(ann_path)
    
    def search(self,query,topk=5):
        sbert_embeds = self.sbert.encode(query)
        results = self.ann.get_nns_by_vector(sbert_embeds, topk, search_k=-1, include_distances=False)
        for res in results:
            id = self.embeddings_dict["model_id_to_paper_id"][res]
            out = self.json_data[str(id)]
            print(res,out["title"])
            webbrowser.open_new_tab(out["url"])
            print("=="*70)
        return results


if __name__ == "__main__":
    annoy_idxr = AnnoyIndexer(
        embeddings_path="Chapter - 8 : Scraping/dataset/data_embeddings.pt",
        data_path="Chapter - 8 : Scraping/scraped_data/neurips2022.json",
        model_str="pritamdeka/S-Scibert-snli-multinli-stsb"
    )
    out = annoy_idxr.search(input("Query    : "))
    print()