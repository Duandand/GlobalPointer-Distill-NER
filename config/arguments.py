# coding=utf-8
from argparse import ArgumentParser

parser = ArgumentParser(description="common-ner")
parser.add_argument(
    "--cmd",
    default=None,
    type=str,
    choices=["fine_tune", "test"],
    help="Command to control training, and the options are fine_tune or test",
)

# dir
parser.add_argument(
    "--dir_training_data",
    default=None,
    type=str,
    help="Path for traing data",
)
parser.add_argument(
    "--dir_checkpoint",
    default=None,
    type=str,
    help="Save the model path to checkpoint",
)
parser.add_argument(
    "--dir_init_checkpoint",
    default=None,
    type=str,
    help="Initialize the BERT model parameters",
)
parser.add_argument(
    "--dir_summary",
    default=None,
    type=str,
    help="Path to save tensorboard logs",
)
parser.add_argument("--dir_log", default=None, type=str, help="Path for saving logs")
parser.add_argument("--dir_ent2id", default=None, type=str, help="Path for ent2id")

parser.add_argument("--vocab_path", default=None, type=str, help="Path for bert vocab.")

parser.add_argument(
    "--file_name_train",
    default=None,
    type=str,
    help="training set file name",
)
parser.add_argument(
    "--file_name_dev",
    default=None,
    type=str,
    help="dev set file name",
)
parser.add_argument(
    "--file_name_test", default=None, type=str, help="test set file name."
)


# multi-process
parser.add_argument(
    "--local_rank", type=int, help="For distributed training: local_rank"
)
parser.add_argument(
    "--no_cuda", action="store_true", default=False, help="Is there a cuda"
)


# training related
parser.add_argument(
    "--ent_type_size",
    default=None,
    type=int,
    help="Total type of entity for bert-global-pointer.",
)
parser.add_argument("--seed", default=None, type=int, help="Random seed for training")

parser.add_argument("--map_location", default=None, type=str, help="CPU or GPU device")

parser.add_argument(
    "--weight_decay_rate",
    default=None,
    type=float,
    help="Weight decay if we apply some.",
)
parser.add_argument(
    "--learning_rate",
    default=None,
    type=float,
    help="The initial learning rate for Adam.",
)
parser.add_argument(
    "--warmup_steps", default=None, type=float, help="Total number of warm up steps."
)
parser.add_argument(
    "--num_train_steps",
    default=None,
    type=int,
    help="Total number of training steps to perform.",
)
parser.add_argument(
    "--num_train_epochs",
    default=None,
    type=int,
    help="Total number of training epochs to perform.",
)
parser.add_argument(
    "--gradient_accumulation_steps",
    default=None,
    type=int,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument(
    "--batch_size",
    default=None,
    type=int,
    help="This is the maximum batch size set."
    "When it is 32 and gradient_accumulation_steps is 4, then batch_size for each is 32//4=8.",
)
parser.add_argument(
    "--save_checkpoint_interval",
    default=None,
    type=int,
    help="Save checkpoint every X updates steps.",
)

# bert-model related
parser.add_argument(
    "--vocab_size", default=None, type=int, help="voacab size of bert embedding."
)
parser.add_argument(
    "--hidden_size",
    default=None,
    type=int,
    help="hidden size of bert embedding layer/ bert pooler/ bert model/ lebert.",
)
parser.add_argument(
    "--intermediate_size", default=None, type=int, help="intermediate size of bert."
)
parser.add_argument(
    "--hidden_act", default=None, type=str, help="hidden activation of bert model."
)
parser.add_argument(
    "--max_position_embeddings",
    default=None,
    type=int,
    help="max position embeddings of bert embedding layer.",
)
parser.add_argument(
    "--do_random_next",
    default=None,
    type=bool,
    help="do_random_next of bert embedding layer/ bert model.",
)
parser.add_argument(
    "--type_vocab_size",
    default=None,
    type=int,
    help="type vocab size of bert embedding.",
)
parser.add_argument(
    "--layer_norm_eps",
    default=None,
    type=float,
    help="layer norm eps of bert embedding.",
)
parser.add_argument(
    "--hidden_dropout_prob",
    default=None,
    type=float,
    help="hidden dropout prob of bert embedding layer.",
)
parser.add_argument(
    "--attention_probs_dropout_prob",
    default=None,
    type=float,
    help="attention probs dropout prob of bert self-attention",
)
# bert model encoder
parser.add_argument(
    "--output_attentions",
    action="store_true",
    default=False,
    help="output attentions of bert model encoder.",
)
parser.add_argument(
    "--output_hidden_states",
    action="store_true",
    default=False,
    help="output hidden states of bert model encoder.",
)
parser.add_argument(
    "--num_attention_heads",
    default=None,
    type=int,
    help="num attention heads of bert self-attention",
)
parser.add_argument(
    "--num_hidden_layers",
    default=None,
    type=int,
    help="num hidden layers of bert model encoder.",
)
parser.add_argument(
    "--use_checkpoint_sequential",
    default=None,
    type=bool,
    help="Apply Checkpoint to a model built for Sequential.",
)
parser.add_argument(
    "--chunks", default=None, type=int, help="Used with the use checkpoint sequential."
)
parser.add_argument(
    "--initializer_range", default=None, type=int, help="Initialize the range."
)

# TinyBERT
parser.add_argument(
    "--temperature", default=None, type=float, help="temperature for TinyBERT."
)
parser.add_argument(
    "--intermediate_size_tinybert",
    default=None,
    type=int,
    help="intermediate size of tinybert.",
)
parser.add_argument(
    "--hidden_size_tinybert",
    default=None,
    type=int,
    help="hidden size of tinybert embedding layer/ bert pooler/ bert model/ lebert.",
)
parser.add_argument(
    "--num_hidden_layers_tinybert",
    default=None,
    type=int,
    help="num hidden layers of tinybert model encoder.",
)

parser.add_argument(
    "--learning_rate_tinybert_one",
    default=None,
    type=float,
    help="learning rate for tinybert trans distill.",
)
parser.add_argument(
    "--learning_rate_tinybert_two",
    default=None,
    type=float,
    help="learning rate for tinybert pred distill.",
)
parser.add_argument(
    "--num_train_epochs_tinybert_one",
    default=None,
    type=int,
    help="number of train epochs for tinybert trans distill.",
)
parser.add_argument(
    "--num_train_epochs_tinybert_two",
    default=None,
    type=int,
    help="number of train epochs for tinybert pred distill.",
)
