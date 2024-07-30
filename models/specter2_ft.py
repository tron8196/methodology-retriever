from torch import nn
from transformers import AutoModel


class Specter2Ft(nn.Module):
    def __init__(self, specter2_model_name, freeze_encoder=True):
        super(Specter2Ft, self).__init__()
        self.specter2_model = AutoModel.from_pretrained(specter2_model_name)
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
    
    def triplet_forward(self, anchor, positive, negative):
        anchor_embeds = self.specter2_model(**anchor).last_hidden_state[:, 0, :]
        positive_embeds = self.specter2_model(**positive).last_hidden_state[:, 0, :]
        negative_embeds = self.specter2_model(**negative).last_hidden_state[:, 0, :]
        loss =  self.triplet_loss(anchor_embeds, positive_embeds, negative_embeds)
        return {"loss": loss}


    def forward(self, citing, cited, text, citing_cited_negative, cited_negative, citing_negative):

        #citing cited triplet
        citing_cited_loss = self.triplet_forward(citing, cited, citing_cited_negative)

        #text cited triplet
        text_cited_liss = self.triplet_forward(text, cited, cited_negative)

        #text citing triplet
        text_citing_loss = self.triplet_forward(text, citing, citing_negative)

        total_loss = citing_cited_loss["loss"] + text_cited_liss["loss"] + text_citing_loss["loss"]

        return {"loss": total_loss}



