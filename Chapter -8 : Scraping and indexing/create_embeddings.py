from sentence_transformers import SentenceTransformer
import torch
import json 

def load_data(data_path):
    json_data = json.load(open(data_path,"r"))
    paper_ids = []
    paper_texts = []
    for k,v in json_data.items():
        paper_ids.append(k)
        paper_texts.append( v["title"].strip()+". "+v["abstract"] )
    return paper_ids,paper_texts

def save_embeddings(data_path,model_str:str='all-MiniLM-L6-v2',save_embs_path=None):
    ids,texts = load_data(data_path)
    sbert = SentenceTransformer(model_str)
    sbert_embeds = sbert.encode(texts,batch_size=64*2,
                            show_progress_bar=True,
                            device="cuda" if torch.cuda.is_available() else "cpu"
            )
    if save_embs_path is not None:
        res_dict = {}
        model_id_to_paper_id = {}
        for idx,(k,v) in enumerate(zip(ids,sbert_embeds)):
            model_id_to_paper_id[idx] = k
            res_dict[idx] = v


        torch.save(
            {
                "embeddings":res_dict,
                "model_str":model_str,
                "embed_size":sbert_embeds.shape[1],
                "model_id_to_paper_id":model_id_to_paper_id
            },
            save_embs_path
        )
    print("Done !")
        
if __name__=="__main__":
    save_embeddings(
        data_path="Chapter - 8 : Scraping/scraped_data/neurips2022.json",
        model_str="pritamdeka/S-Scibert-snli-multinli-stsb",
        save_embs_path="Chapter - 8 : Scraping/dataset/data_embeddings.pt"
    )

        