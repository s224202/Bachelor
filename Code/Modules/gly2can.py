from transformers import AutoTokenizer, EncoderDecoderModel, BertConfig, EncoderDecoderConfig
class gly2can():
    """
    A class to represent a glycan translation model.
    """
    def __init__(self, orig_nomen:str, target_nomen:str):
        self.orig_nomen = orig_nomen
        self.target_nomen = target_nomen
        config_encoder = BertConfig()
        config_decoder = BertConfig(is_decoder=True)
        config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
        self.model = EncoderDecoderModel(config=config)
        self.model.config.decoder_start_token_id = 234
        self.model.config.pad_token_id = 0
        self.model.config.eos_token_id = 234
        self.model.config.bos_token_id = 234
        

def glycan_tokenizer(glycan_sequences):
    base_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased")
    ## quick sanity check prints to make sure this pretraining actually helps
    #print(base_tokenizer.tokenize(glycan_sequences[2]))
    glycan_tok = base_tokenizer.train_new_from_iterator(get_training_corpus(glycan_sequences=glycan_sequences),30522)
    glycan_tok.add_tokens(['[Glycan]', '[Glycan]'])
    glycan_tok.add_special_tokens({
        'additional_special_tokens': ['[Glycan]', '[Glycan]']
    })
    glycan_tok.pad_token = '[PAD]'
    glycan_tok.bos_token = '[Glycan]'
    glycan_tok.eos_token = '[Glycan]'
    #print(glycan_tok.tokenize(glycan_sequences[2]))
    return glycan_tok

def get_training_corpus(glycan_sequences):
    return (
        glycan_sequences[i:1+100]
        for i in range(0, len(glycan_sequences[0]), 100)
    )