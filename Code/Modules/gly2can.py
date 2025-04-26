import transformers.models.encoder_decoder as encoder_decoder
from transformers import AutoTokenizer
class gly2can(encoder_decoder.EncoderDecoderModel):
    """
    A class to represent a glycan translation model.
    Inherits from the EncoderDecoderModel class of the transformers library.
    """
    def __init__(self, encoder, decoder):
        super(gly2can, self).__init__(encoder=encoder, decoder=decoder)
        self.encoder = encoder
        self.decoder = decoder

def glycan_tokenizer(glycan_sequences):
    base_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased")
    ## quick sanity check prints to make sure this pretraining actually helps
    print(base_tokenizer.tokenize(glycan_sequences[2]))
    glycan_tok = base_tokenizer.train_new_from_iterator(get_training_corpus(glycan_sequences=glycan_sequences), 5000)
    print(glycan_tok.tokenize(glycan_sequences[2]))
    return glycan_tok

def get_training_corpus(glycan_sequences):
    return (
        glycan_sequences[i:1+100]
        for i in range(0, len(glycan_sequences[0]), 100)
    )