import torch
from torch.utils.data import Dataset
import random
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm 
import torch.nn.functional as F
from models.specter2_ft import Specter2Ft
from transformers import Trainer, TrainingArguments

class TripletDataCollator:
    def __call__(self, batch):
        citing = [item['citing'] for item in batch]
        cited = [item['cited'] for item in batch]
        text = [item['text'] for item in batch]
        citing_cited_negative = [item['citing_cited_negative'] for item in batch]
        cited_negative = [item['cited_negative'] for item in batch]
        citing_negative = [item['citing_negative'] for item in batch]

        return {
            'citing': self.stack_inputs(citing),
            'cited': self.stack_inputs(cited),
            'text': self.stack_inputs(text),
            'citing_cited_negative': self.stack_inputs(citing_cited_negative),
            'cited_negative': self.stack_inputs(cited_negative),
            'citing_negative': self.stack_inputs(citing_negative)
        }

    def stack_inputs(self, inputs):
        return {
            key: torch.stack([item[key].squeeze(0) for item in inputs])
            for key in inputs[0].keys()
        }

class CitationTripletDataset(Dataset):
    def __init__(self, df_path, specter2_model_name):
        self.data = pd.read_csv(df_path)
        # Load SPECTER2 model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(specter2_model_name, use_fast=False)
        self.model = AutoModel.from_pretrained(specter2_model_name)
        # self.model.load_adapter("allenai/specter2", load_as="specter2", set_active=True)
        self.model.eval()  # Set to evaluation mode
        self.cited_paper_candidates = list(set(self.data.cited_paper_abstract.to_list()))
        self.cited_paper_embeds = self.get_embedddings(self.data, col="cited_paper_abstract")
        self.citing_paper_embeds = self.get_embedddings(self.data, col="citing_paper_abstract")

    def get_embedddings(self, df, col="cited_paper_abstract"):
        embeddings = []
        for abstract in tqdm(df[col]):
            inputs = self.tokenizer(abstract, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings.append(outputs.last_hidden_state[:, 0, :])
        return torch.stack(embeddings).squeeze()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        citing_abstract, cited_abstract, citation_text = self.get_data_row(idx)

        #citing paper (anchor) and cited paper(positive) tuple
        negative_cited_type1 = self.get_negative_type1(idx)
        negative_cited_type2 = self.get_negative_type2(cited_abstract, self.cited_paper_embeds)

        #citation text and cited paper tuple
        negative_citing_abstract = self.get_negative_type2(citing_abstract, self.citing_paper_embeds)


        return {
            'citing': self.text_to_tensor(citing_abstract),
            'cited': self.text_to_tensor(cited_abstract),
            'text': self.text_to_tensor(citation_text) ,
            'citing_cited_negative': self.text_to_tensor(random.choice([negative_cited_type1, negative_cited_type2])),
            'cited_negative': self.text_to_tensor(negative_cited_type2),
            'citing_negative': self.text_to_tensor(negative_citing_abstract)
            }

    def get_data_row(self, idx):
        citing_abstract = self.data.iloc[idx]["citing_paper_abstract"]
        cited_abstract = self.data.iloc[idx]["cited_paper_abstract"]
        citation_text = self.data.iloc[idx]["citation_text"]
        return citing_abstract, cited_abstract, citation_text

    def get_negative_type1(self, idx):
        citing_abstract =  self.data.iloc[idx]["citing_paper_abstract"]
        df_sample = self.data[(self.data["citing_paper_abstract"] == citing_abstract) & (self.data["intent"] == 0)]
        if len(df_sample) > 0:
            return df_sample.sample(1).iloc[0]["cited_paper_abstract"]
        else:
            return ""

    def get_negative_type2(self, positive, candidate_embeds):        
        inp = self.tokenizer(positive, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        positive_embed = self.model(**inp).last_hidden_state[:, 0, :]
        similarities = candidate_embeds@positive_embed.T
        # Get top k most similar items
        k = 5  # Number of top items to consider
        top_k_similarities, top_k_indices = torch.topk(similarities, k, dim=0)
        # Convert similarities to probabilities using softmax
        probabilities = F.softmax(top_k_similarities, dim=0)

        # Ensure probabilities is a 1D tensor
        probabilities = probabilities.view(-1)

        # Sample an index from the top k subset
        sampled_index = torch.multinomial(probabilities, num_samples=1).item()
        # Get the selected ID
        selected_id = top_k_indices[sampled_index].item()

        return self.data.iloc[selected_id]["cited_paper_abstract"]        

    def get_specter2_embeddings(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # Use [CLS] token embedding

    def text_to_tensor(self, text):
        return self.tokenizer(text, padding='max_length', truncation=True, 
                              max_length=512, return_tensors="pt")


