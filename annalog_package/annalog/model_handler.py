import torch
import pickle
from torchtext.legacy.data import Field
from .model import seq2seq_attention, vocabulary

class SMILESModelHandler:
    """
    A handler to initialize and manage the SMILES generation model.
    """
    def __init__(self, src_vocab_path, trg_vocab_path, model_path, device='cuda', max_length=102):
        """
        Initialize the model handler.

        Args:
            src_vocab_path (str): Path to the source vocabulary pickle file.
            trg_vocab_path (str): Path to the target vocabulary pickle file.
            model_path (str): Path to the trained model file.
            device (str): Device type ('cuda' or 'cpu').
            max_length (int): Maximum length for SMILES sequences.
        """
        self.device = torch.device("cuda" if device == 'cuda' and torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        # Initialize fields
        self.SRC = Field(tokenize=vocabulary.SMILESTokenizer().tokenize,
                         init_token='<sos>', eos_token='<eos>',
                         lower=False, batch_first=True)
        self.TRG = Field(tokenize=vocabulary.SMILESTokenizer().tokenize,
                         init_token='<sos>', eos_token='<eos>',
                         lower=False, batch_first=True)

        # Load vocabularies
        with open(src_vocab_path, 'rb') as f:
            self.SRC.vocab = pickle.load(f)
        with open(trg_vocab_path, 'rb') as f:
            self.TRG.vocab = pickle.load(f)

        # Model parameters
        INPUT_DIM = len(self.SRC.vocab)
        OUTPUT_DIM = len(self.TRG.vocab)
        HID_DIM = 256
        ENC_LAYERS = 3
        DEC_LAYERS = 3
        ENC_HEADS = 8
        DEC_HEADS = 8
        ENC_PF_DIM = 512
        DEC_PF_DIM = 512
        ENC_DROPOUT = 0.1
        DEC_DROPOUT = 0.1

        # Initialize encoder and decoder
        enc = seq2seq_attention.Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, self.device, max_length)
        dec = seq2seq_attention.Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, self.device, max_length)

        # Initialize model
        SRC_PAD_IDX = self.SRC.vocab.stoi[self.SRC.pad_token]
        TRG_PAD_IDX = self.TRG.vocab.stoi[self.TRG.pad_token]
        self.model = seq2seq_attention.Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, self.device).to(self.device)

        # Load pre-trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def get_model_and_fields(self):
        """
        Retrieve the initialized model, SRC and TRG fields, and device.

        Returns:
            tuple: (SRC, TRG, model, device)
        """
        return self.SRC, self.TRG, self.model, self.device