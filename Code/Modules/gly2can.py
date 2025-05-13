from transformers import AutoTokenizer, EncoderDecoderModel, BertConfig, EncoderDecoderConfig, BertGenerationConfig, BertGenerationEncoder, BertGenerationDecoder
class gly2can():
    """
    A class to represent a glycan translation model.
    """
    def __init__(self, orig_nomen:str, target_nomen:str, load_model:bool=False):
        self.orig_nomen = orig_nomen
        self.target_nomen = target_nomen
        if load_model:
            self.model = EncoderDecoderModel.from_pretrained(f"./Models/{orig_nomen}_{target_nomen}_fine_tuned")
        else:
            config = BertGenerationConfig.from_pretrained("google-bert/bert-large-uncased")
            encoder = BertGenerationEncoder(config)
            decoder_config = BertConfig.from_pretrained("google-bert/bert-large-uncased")
            decoder_config.is_decoder = True
            decoder_config.add_cross_attention = True
            decoder = BertGenerationDecoder(decoder_config)
            self.model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
            self.model.config.decoder_start_token_id = 270
            self.model.config.eos_token_id = 270
            self.model.config.pad_token_id = 0

def glycan_tokenizer(glycan_sequences):
    base_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased")
    ## quick sanity check prints to make sure this pretraining actually helps
    #print(base_tokenizer.tokenize(glycan_sequences[2]))
    glycan_tok = base_tokenizer.train_new_from_iterator(get_training_corpus(glycan_sequences=glycan_sequences),30522)
    glycan_tok.add_tokens(['[Glycan]', '[Glycan]'])
    glycan_tok.add_special_tokens({
        'additional_special_tokens': ['[Glycan]', '[Glycan]']
    })
    glycan_tok.model_max_length = 512
    glycan_tok.pad_token = '[PAD]'
    glycan_tok.bos_token = '[Glycan]'
    glycan_tok.eos_token = '[Glycan]'
    print(glycan_tok.pad_token_id)
    print(glycan_tok.bos_token_id)
    print(glycan_tok.eos_token_id)
    #print(glycan_tok.tokenize(glycan_sequences[2]))
    return glycan_tok

def get_training_corpus(glycan_sequences):
    return (
        glycan_sequences[i:1+100]
        for i in range(0, len(glycan_sequences[0]), 100)
    )