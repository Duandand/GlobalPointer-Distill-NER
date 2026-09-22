# coding=utf-8
import json
import os


class HyperParams(object):
    def __init__(self):
        self.cmd = "fine_tune"
        # dir（默认相对项目根目录，即 config/ 的上一级；可被 CLI 参数覆盖）
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dir_training_data = os.path.join(root_dir, "data/ner_data")
        self.dir_checkpoint = os.path.join(root_dir, "checkpoint/NER_GPCommonNer")
        self.dir_init_checkpoint = os.path.join(
            root_dir, "init_ckpt/pytorch_model.bin"
        )
        self.dir_summary = os.path.join(root_dir, "summary/NER_GPCommonNer")
        self.dir_log = os.path.join(root_dir, "logs")
        self.dir_ent2id = os.path.join(root_dir, "resources/ner_ent2id.json")
        self.vocab_path = os.path.join(root_dir, "resources/vocab.txt")

        # file
        self.file_name_train = "train_ner"
        self.file_name_dev = "dev_ner"
        self.file_name_test = "test_ner"

        # multi-process
        self.local_rank = -1
        self.no_cuda = False  # 默认是false

        # train
        self.ent_type_size = 4  # bert-gp 实体类型数
        # common
        self.seed = 42
        self.map_location = "cpu"
        self.weight_decay_rate = 0.01
        self.learning_rate = 3e-5
        self.warmup_steps = 0.1
        self.num_train_steps = None  # 和num_train_epochs只使用一个
        self.num_train_epochs = 1
        self.gradient_accumulation_steps = 1
        self.batch_size = 32
        self.save_checkpoint_interval = 50  # 每50次评估并保存一次模型

        # bert model
        self.vocab_size = 21128
        self.hidden_size = 768
        self.intermediate_size = 3072
        self.hidden_act = "gelu"
        self.max_position_embeddings = 512
        self.do_random_next = False
        self.type_vocab_size = 2
        self.layer_norm_eps = 1e-12
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.output_attentions = True  # bert model是否输出attention
        self.output_hidden_states = True  # bert_model是否输出hidden
        self.num_attention_heads = 12
        self.num_hidden_layers = 12
        self.use_checkpoint_sequential = False
        self.chunks = 12
        self.initializer_range = 0.02

        # TinyBERT
        self.temperature = 1.0
        self.intermediate_size_tinybert = 1200
        self.hidden_size_tinybert = 312
        self.num_hidden_layers_tinybert = 4
        self.learning_rate_tinybert_one = 5e-05
        self.num_train_epochs_tinybert_one = 1
        self.learning_rate_tinybert_two = 3e-05
        self.num_train_epochs_tinybert_two = 1

    @classmethod
    def init_from_parsed_args(cls, parsed_args):
        hp = cls()
        args_dict = vars(parsed_args)
        for k in args_dict:
            if args_dict[k]:
                setattr(hp, k, args_dict[k])  # 使用arguments里的参数值覆盖掉HyperParams里的
        return hp

    def to_json_string(self):
        return json.dumps(self.__dict__, indent=4)
