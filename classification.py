from util import *
import numpy as np
import random
from transformers import XLMTokenizer, XLMWithLMHeadModel, XLMModel


def get_similarity_score(sentence_emb, label_embs):
    sentence_emb = sentence_emb.view(1, -1)  # n*1024
    tmp_type_embeddings = torch.mean(torch.cat(label_embs, dim=0), dim=0).view(1, -1)  # n*1024

    similarities = cos(sentence_emb, tmp_type_embeddings).view(1, -1)  # 1*n
    return similarities.tolist()[0][0]  # [1,1]


def print_performance():
    print("Predicate HIT@1: {}, {}/{}".format(predicate_top1_cnt / predicate_total_cnt, predicate_top1_cnt,
                                              predicate_total_cnt))
    print("Predicate HIT@3: {}, {}/{}".format(predicate_top3_cnt / predicate_total_cnt, predicate_top3_cnt,
                                              predicate_total_cnt))
    print("Predicate HIT@5: {}, {}/{}".format(predicate_top5_cnt / predicate_total_cnt, predicate_top5_cnt,
                                              predicate_total_cnt))

    print("Argument HIT@1: {}, {}/{}".format(argument_top1_cnt / argument_total_cnt, argument_top1_cnt,
                                             argument_total_cnt))
    print("Argument HIT@3: {}, {}/{}".format(argument_top3_cnt / argument_total_cnt, argument_top3_cnt,
                                             argument_total_cnt))
    print("Argument HIT@5: {}, {}/{}".format(argument_top5_cnt / argument_total_cnt, argument_top5_cnt,
                                             argument_total_cnt))


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default='7', type=str, required=False,
                    help="choose which gpu to use")
parser.add_argument("--representation_source", default='nyt', type=str, required=False,
                    help="choose which gpu to use")
parser.add_argument("--model", default='bert-large', type=str, required=False,
                    help="choose which gpu to use")
parser.add_argument("--pooling_method", default='final', type=str, required=False,
                    help="choose which gpu to use")
parser.add_argument("--weight", default=10, type=float, required=False,
                    help="weight assigned to triggers")
parser.add_argument("--num_anchor", default=10, type=int, required=False,
                    help="weight assigned to triggers")

# setup your own GPU here
args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
print('current device:', device)

train_data = list()
with open('data_ACE2005/train.oneie.json', 'r') as f:
    for line in f:
        train_data.append(json.loads(line))

dev_data = list()
with open('data_ACE2005/dev.oneie.json', 'r') as f:
    for line in f:
        dev_data.append(json.loads(line))

test_data = list()
with open('data_ACE2005/test.oneie.json', 'r') as f:
    for line in f:
        test_data.append(json.loads(line))
all_data = train_data + dev_data + test_data
if args.model == 'roberta-large':
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    model = RobertaModel.from_pretrained('roberta-large').to(device)
elif args.model == 'roberta-base':
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-large').to(device)
elif args.model == 'bert-large':
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    model = BertModel.from_pretrained('bert-large-uncased').to(device)
elif args.model == 'mbert':
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    model = BertModel.from_pretrained('bert-base-multilingual-uncased').to(device)
elif args.model == 'xlnet':
    tokenizer = XLMTokenizer.from_pretrained("xlm-mlm-tlm-xnli15-1024")
    model = XLMModel.from_pretrained("xlm-mlm-tlm-xnli15-1024")
else:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
model.eval()
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-20)

etype_to_distinct_embeddings = dict()
rtype_to_distinct_embeddings = dict()

print('start to load the reference data...')
with open('data/all_verb_reference_sentences.json', 'r', encoding='utf-8') as f:
    verb_keyword_to_sentences = json.load(f)

with open('data/all_noun_reference_sentences.json', 'r', encoding='utf-8') as f:
    noun_keyword_to_sentences = json.load(f)

with open('data/all_role_reference_sentences.json', 'r', encoding='utf-8') as f:
    all_role_keyword_to_sentences = json.load(f)

for tmp_e_type in tqdm(event_types, desc='Loading predicate embeddings'):
    etype_to_distinct_embeddings[tmp_e_type] = list()
    for tmp_w in verb_keywords[tmp_e_type]:
        random.shuffle(verb_keyword_to_sentences[tmp_w])
        for tmp_example in verb_keyword_to_sentences[tmp_w][:args.num_anchor]:
            sent_emb = get_represetation(tmp_example[0], (tmp_example[1], tmp_example[2]),
                                         tokenizer, model, device)
            if not isinstance(sent_emb, str):
                etype_to_distinct_embeddings[tmp_e_type].append(sent_emb)
    for tmp_w in noun_keywords[tmp_e_type]:
        random.shuffle(noun_keyword_to_sentences[tmp_w])
        for tmp_example in noun_keyword_to_sentences[tmp_w][:args.num_anchor]:
            sent_emb = get_represetation(tmp_example[0], (tmp_example[1], tmp_example[2]),
                                         tokenizer, model, device)
            if not isinstance(sent_emb, str):
                etype_to_distinct_embeddings[tmp_e_type].append(sent_emb)
for tmp_r_type in tqdm(role_types, desc='Loading argument embeddings'):
    rtype_to_distinct_embeddings[tmp_r_type] = list()
    for tmp_n in role_keywords[tmp_r_type]:
        random.shuffle(all_role_keyword_to_sentences[tmp_n])
        selected_sentences = all_role_keyword_to_sentences[tmp_n][:args.num_anchor+1]
        for tmp_example in selected_sentences:
            sent_emb = get_represetation(tmp_example[0], (tmp_example[1], tmp_example[2]),
                                         tokenizer, model, device, representation_type='mask')
            if not isinstance(sent_emb, str):
                    rtype_to_distinct_embeddings[tmp_r_type].append(sent_emb)
            if len(rtype_to_distinct_embeddings[tmp_r_type]) >= args.num_anchor:
                break

predicate_top1_cnt = 0.0
predicate_top3_cnt = 0.0
predicate_top5_cnt = 0.0
predicate_total_cnt = 0.0

argument_top1_cnt = 0.0
argument_top3_cnt = 0.0
argument_top5_cnt = 0.0
argument_total_cnt = 0.0

chosen_event_types = event_types
chosen_role_types = role_types

model.eval()
for sample in tqdm(dev_data, desc="Evaluation : " + str(args.weight)):
    entity_id_to_details = dict()
    for tmp_entity in sample['entity_mentions']:
        entity_id_to_details[tmp_entity['id']] = tmp_entity
    for event in sample['event_mentions']:
        if event['event_type'] not in chosen_event_types:
            continue
        sent_emb = get_represetation(sample['tokens'], (event['trigger']['start'], event['trigger']['end']),
                                     tokenizer, model, device)
        predicate_score = list()
        for tmp_etype in chosen_event_types:
            predicate_score.append(
                get_similarity_score(sent_emb, etype_to_distinct_embeddings[tmp_etype]))
        argument_scores = list()
        entity_types = list()
        for tmp_argument in event['arguments']:
            sent_emb = get_represetation(sample['tokens'], (entity_id_to_details[tmp_argument['entity_id']]['start'], entity_id_to_details[tmp_argument['entity_id']]['end']),
                                         tokenizer, model, device, representation_type='mask')
            tmp_scores = list()
            for tmp_r_type in chosen_role_types:
                tmp_scores.append(get_similarity_score(sent_emb, rtype_to_distinct_embeddings[tmp_r_type]))
            argument_scores.append(tmp_scores)
            entity_types.append(entity_id_to_details[tmp_argument['entity_id']]['entity_type'])
        if len(argument_scores) > 0:
            tmp_optimizer = gurobi_opt(predicate_score, argument_scores, entity_types, chosen_event_types,
                                       chosen_role_types, weight=args.weight)
            optimized_predicates, optimized_arguments = tmp_optimizer.optimize_all()
            if event['event_type'] in optimized_predicates[:1]:
                predicate_top1_cnt += 1
            if event['event_type'] in optimized_predicates[:3]:
                predicate_top3_cnt += 1
            if event['event_type'] in optimized_predicates[:5]:
                predicate_top5_cnt += 1
            predicate_total_cnt += 1

            for i, tmp_argument in enumerate(event['arguments']):
                if tmp_argument['role'] in optimized_arguments[i][:1]:
                    argument_top1_cnt += 1
                if tmp_argument['role'] in optimized_arguments[i][:3]:
                    argument_top3_cnt += 1
                if tmp_argument['role'] in optimized_arguments[i][:5]:
                    argument_top5_cnt += 1
                argument_total_cnt += 1
        else:
            scores = list()
            for tmp_score in predicate_score:
                scores.append(torch.tensor(tmp_score).view(1, -1).to(device))
            sorted_similarities, argument_indexes = torch.sort(torch.cat(scores, dim=1), dim=1, descending=True)
            sorted_etypes = list()
            for tmp_position in argument_indexes.tolist()[0]:
                sorted_etypes.append(chosen_event_types[tmp_position])
            if event['event_type'] in sorted_etypes[:1]:
                predicate_top1_cnt += 1
            if event['event_type'] in sorted_etypes[:3]:
                predicate_top3_cnt += 1
            if event['event_type'] in sorted_etypes[:5]:
                predicate_top5_cnt += 1
            predicate_total_cnt += 1
            for tmp_pos, tmp_argument in enumerate(event['arguments']):
                scores = list()
                for tmp_score in argument_scores[tmp_pos]:
                    scores.append(torch.tensor(tmp_score).view(1, -1).to(device))
                sorted_similarities, argument_indexes = torch.sort(torch.cat(scores, dim=1), dim=1, descending=True)
                sorted_rtypes = list()
                for tmp_position in argument_indexes.tolist()[0]:
                    sorted_rtypes.append(chosen_role_types[tmp_position])
                if tmp_argument['role'] in sorted_rtypes[:1]:
                    argument_top1_cnt += 1
                if tmp_argument['role'] in sorted_rtypes[:3]:
                    argument_top3_cnt += 1
                if tmp_argument['role'] in sorted_rtypes[:5]:
                    argument_top5_cnt += 1
                argument_total_cnt += 1

print_performance()

