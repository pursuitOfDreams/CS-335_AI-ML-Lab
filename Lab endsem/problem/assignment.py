import numpy as np
import torch
import pickle as pkl
import torch.nn as nn
# sklearn is not allowed strictly

def score(queries_embed:torch.Tensor, corpus_embed:torch.Tensor) -> torch.Tensor:
    """Computes the similarity score between each query and corpus pair.
    Since this function will be autograded, make sure that you stick to the formula mentioned.

    Args:
        queries_embed (torch.Tensor): This is a Tensor of shape (num_queries, embedding_dim)
        corpus_embed (torch.Tensor): This is a Tensor of shape (num_corpus, embedding_dim)

    Returns:
        torch.Tensor: This is a tensor of shape (num_queries, num_corpus)
    """
    scores = torch.zeros(queries_embed.shape[0], corpus_embed.shape[0])
    ### TODO

    ### END TODO
    assert scores.shape == (queries_embed.shape[0], corpus_embed.shape[0])
    return scores

def ranking_loss(scores: torch.Tensor, ground_truth:torch.Tensor, margin:float) -> float:
    """Implements the ranking loss formula mentioned in the Question paper.

    Args:
        scores (torch.Tensor): This is Tensor of predicted similarity scores of shape (num_queries, num_corpus)
        ground_truth (torch.Tensor): This is Tensor of ground truth binary labels 
                            of shape (num_queries, num_corpus). Label 1 means the corresponding pair is relevant
        margin (float): This is a floating point value.

    Returns:
        torch.tensor: The computed Loss. This should be torch.tensor which allows autograd.
    """
    loss = torch.tensor(0)

    ## TODO

    ## End TODO

    assert len(loss.shape) == 0
    return loss
    
def average_precision(scores:torch.Tensor, ground_truth:torch.Tensor) -> float:
    """Implements the average precision formula mentioned in the Question paper.
    Since this function will be autograded, make sure that you stick to the formukla mentioned.

    Args:
        scores (torch.Tensor): This is Tensor of predicted similarity scores of shape (num_queries, num_corpus)
        ground_truth (torch.Tensor): This is Tensor of ground truth binary labels 
                            of shape (num_queries, num_corpus). Label 1 means the corresponding pair is relevant

    Returns:
        float: The computed evaluation metric \in [0,1].
    """
    avg_precision = float(0)
    ## TODO

    ## END TODO
    assert isinstance(avg_precision, float) == True
    return avg_precision

class Model(torch.nn.Module):
    def __init__(self, inp_features=5, out_features=5, margin:float=0.1, *args, **kwargs):
        
        self.inp_features = inp_features
        self.out_features = out_features

        # Define the set embedding parameters here
        ## TODO

        ## End TODO

    def set_embed(self, set_items:torch.Tensor) -> torch.Tensor:
        """Embeds the set of items in a query/corpus

        Args:
            set_items (torch.Tensor): This is a tensor of shape (n, self.inp_features) where n is the number of items in query/corpus

        Returns:
            torch.Tensor: This returns the set embedding tensor of shape (1, self.out_features) 
        """
        set_embed = torch.zeros(1, self.out_features)

        ## TODO

        ## End TODO

        assert set_embed.shape == (1, self.out_features)
        return set_embed

    def forward(self, queries:list, corpus:list) -> torch.Tensor:
        """Forward pass over the query X corpus pairs and computes the predicted relevance scores of shape (num_queries, num_corpus)

        Args:
            queries (list): This is a list of n query Tensors. Recall that each query is a set of items. 
                Assume ith query is a set of n_i items. Thus ith item in the queries list is a Tensor of shape (n_i, self.inp_features)  

            corpus (list):  This is a list of m corpus Tensors. Recall that each corpus is a set of variable number of items.   
                Assume ith corpus is a set of m_i items. Thus ith item in the corpus list is a Tensor of shape (m_i, self.inp_features)  

            Note that you will embed each q \in queries and each c \in corpus using set_embed function to a set embedding of shape (1, self.out_features)

        Returns:
            torch.Tensor: Returns the predicted similarity scores of shape (num_queries, num_corpus)
        """

        # Embed all the queries
        query_emb = torch.cat([self.set_embed(entry) for entry in queries]).squeeze() 
        assert query_emb.shape == (len(queries), self.out_features)

        # Embed all the corpus
        corpus_emb = torch.cat([self.set_embed(entry) for entry in corpus]).squeeze()
        assert corpus_emb.shape == (len(corpus), self.out_features)

        # compute the scores between each query/corpus pair
        scores = score(queries_embed=query_emb, corpus_embed=corpus_emb)
        assert scores.shape == (len(queries), len(corpus))

        return scores

         
if __name__ == "__main__":

    # Load the Dataset
    with open("dataset.pkl", "rb") as file:
        trn_queries, corpus, tst_queries, trn_ground_truth = pkl.load(file)

    inp_features = 5

    # These are some hyperparameters that you can change
    ## TODO
    
    out_features, margin = 5, 0.1

    ## End TODO    
    
    # Define the Model
    model = Model(inp_features=inp_features, out_features=out_features, margin=margin)

    # Train your model here
    # You can create any new function/classes that you need outside main as well
    ## TODO

    ## End TODO

    predicted_scores = model.forward(queries=tst_queries, corpus=corpus)

    # You can test the avg_precision code using training data for correctness of your implementation
    # Make sure you do not have compile time bugs

    with open("output.pkl", "wb") as file:
        pkl.dump(predicted_scores, file)
    with open("model.pkl", "wb") as file:
        pkl.dump(model, file)