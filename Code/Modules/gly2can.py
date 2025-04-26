import transformers.models.encoder_decoder as encoder_decoder

class gly2can(encoder_decoder.EncoderDecoderModel):
    """
    A class to represent a glycan translation model.
    Inherits from the EncoderDecoderModel class of the transformers library.
    """
    def __init__(self, encoder, decoder):
        super(gly2can, self).__init__(encoder=encoder, decoder=decoder)
        self.encoder = encoder
        self.decoder = decoder