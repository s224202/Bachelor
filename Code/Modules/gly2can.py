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
            config_encoder = BertConfig()
            ## Add cross attention was not present during training originally because im dumb, but has been added now, in case anyone wants to use my code.
            config_decoder = BertConfig(is_decoder=True, add_cross_attention=True)
            config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
            self.model = EncoderDecoderModel(config=config,)
            self.model.config.decoder_start_token_id = 234
            self.model.config.pad_token_id = 0
            self.model.config.eos_token_id = 234
            self.model.config.bos_token_id = 234

def glycan_tokenizer(glycan_sequences):
    base_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
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