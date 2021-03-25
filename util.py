import torch
import json
from transformers import RobertaTokenizer, RobertaModel, BertTokenizer, BertModel
import numpy as np
import argparse
import os
from tqdm import tqdm
import spacy
from multiprocessing import Pool
from torch.nn.modules.distance import PairwiseDistance
from gurobi import *
from scipy.special import softmax

event_schema_to_roles = dict()
event_schema_to_roles['Business:Declare-Bankruptcy'] = ['Org', 'Place']
event_schema_to_roles['Business:End-Org'] = ['Place', 'Org']
event_schema_to_roles['Business:Merge-Org'] = ['Org']
event_schema_to_roles['Business:Start-Org'] = ['Org', 'Agent', 'Place']
event_schema_to_roles['Conflict:Attack'] = ['Attacker', 'Place', 'Target', 'Instrument', 'Victim', 'Agent']
event_schema_to_roles['Conflict:Demonstrate'] = ['Place', 'Entity']
event_schema_to_roles['Contact:Meet'] = ['Place', 'Entity']
event_schema_to_roles['Contact:Phone-Write'] = ['Entity', 'Place']
event_schema_to_roles['Justice:Acquit'] = ['Defendant', 'Adjudicator']
event_schema_to_roles['Justice:Appeal'] = ['Adjudicator', 'Plaintiff', 'Place']
event_schema_to_roles['Justice:Arrest-Jail'] = ['Person', 'Agent', 'Place']
event_schema_to_roles['Justice:Charge-Indict'] = ['Adjudicator', 'Defendant', 'Prosecutor', 'Place']
event_schema_to_roles['Justice:Convict'] = ['Defendant', 'Adjudicator', 'Place']
event_schema_to_roles['Justice:Execute'] = ['Agent', 'Place', 'Person']
event_schema_to_roles['Justice:Extradite'] = ['Destination', 'Origin', 'Person', 'Agent']
event_schema_to_roles['Justice:Fine'] = ['Entity', 'Adjudicator', 'Place']
event_schema_to_roles['Justice:Pardon'] = ['Defendant', 'Adjudicator', 'Place']
event_schema_to_roles['Justice:Release-Parole'] = ['Person', 'Entity', 'Place']
event_schema_to_roles['Justice:Sentence'] = ['Adjudicator', 'Defendant', 'Place']
event_schema_to_roles['Justice:Sue'] = ['Adjudicator', 'Defendant', 'Plaintiff', 'Place']
event_schema_to_roles['Justice:Trial-Hearing'] = ['Defendant', 'Place', 'Prosecutor', 'Adjudicator']
event_schema_to_roles['Life:Be-Born'] = ['Person', 'Place']
event_schema_to_roles['Life:Die'] = ['Victim', 'Place', 'Agent', 'Instrument', 'Person']
event_schema_to_roles['Life:Divorce'] = ['Person', 'Place']
event_schema_to_roles['Life:Injure'] = ['Victim', 'Instrument', 'Agent', 'Place']
event_schema_to_roles['Life:Marry'] = ['Person', 'Place']
event_schema_to_roles['Movement:Transport'] = ['Destination', 'Artifact', 'Origin', 'Agent', 'Vehicle', 'Victim',
                                               'Place']
event_schema_to_roles['Personnel:Elect'] = ['Person', 'Place', 'Entity']
event_schema_to_roles['Personnel:End-Position'] = ['Person', 'Place', 'Entity']
event_schema_to_roles['Personnel:Nominate'] = ['Person', 'Agent']
event_schema_to_roles['Personnel:Start-Position'] = ['Person', 'Entity', 'Place']
event_schema_to_roles['Transaction:Transfer-Money'] = ['Giver', 'Recipient', 'Beneficiary', 'Place']
event_schema_to_roles['Transaction:Transfer-Ownership'] = ['Beneficiary', 'Seller', 'Artifact', 'Buyer', 'Place']

event_types = ['Business:Declare-Bankruptcy', 'Business:End-Org', 'Business:Merge-Org', 'Business:Start-Org',
               'Conflict:Attack', 'Conflict:Demonstrate', 'Contact:Meet', 'Contact:Phone-Write', 'Justice:Acquit',
               'Justice:Appeal', 'Justice:Arrest-Jail', 'Justice:Charge-Indict', 'Justice:Convict', 'Justice:Execute',
               'Justice:Extradite', 'Justice:Fine', 'Justice:Pardon', 'Justice:Release-Parole', 'Justice:Sentence',
               'Justice:Sue', 'Justice:Trial-Hearing', 'Life:Be-Born', 'Life:Die', 'Life:Divorce', 'Life:Injure',
               'Life:Marry', 'Movement:Transport', 'Personnel:Elect', 'Personnel:End-Position', 'Personnel:Nominate',
               'Personnel:Start-Position', 'Transaction:Transfer-Money', 'Transaction:Transfer-Ownership']

role_types = ['Adjudicator', 'Agent', 'Artifact', 'Attacker', 'Beneficiary', 'Buyer', 'Defendant', 'Destination',
              'Entity', 'Giver', 'Instrument', 'Org', 'Origin', 'Person', 'Place', 'Plaintiff', 'Prosecutor',
              'Recipient', 'Seller', 'Target', 'Vehicle', 'Victim']

event_types_23 = ['Justice:Pardon', 'Justice:Extradite', 'Justice:Acquit', 'Personnel:Nominate', 'Business:Merge-Org',
                  'Justice:Execute', 'Justice:Fine', 'Life:Divorce', 'Business:Declare-Bankruptcy', 'Business:End-Org',
                  'Justice:Release-Parole', 'Justice:Appeal', 'Business:Start-Org', 'Life:Be-Born', 'Justice:Sue',
                  'Justice:Convict', 'Life:Marry', 'Conflict:Demonstrate', 'Justice:Arrest-Jail', 'Justice:Sentence',
                  'Justice:Charge-Indict', 'Justice:Trial-Hearing', 'Personnel:Start-Position']

role_types_23 = ['Person', 'Plaintiff', 'Defendant', 'Place', 'Entity', 'Org', 'Destination', 'Prosecutor',
                 'Agent', 'Origin', 'Adjudicator']

role_to_type = dict()
role_to_type['Adjudicator'] = ['PER', 'ORG', 'GPE', 'NAN', 'MISC', '']
role_to_type['Agent'] = ['ORG', 'PER', 'GPE', 'NAN', 'MISC', '']
role_to_type['Artifact'] = ['PER', 'VEH', 'WEA', 'ORG', 'FAC', 'NAN', 'MISC', '']
role_to_type['Attacker'] = ['PER', 'GPE', 'ORG', 'NAN', 'MISC', '']
role_to_type['Beneficiary'] = ['GPE', 'PER', 'ORG', 'NAN', 'MISC', '']
role_to_type['Buyer'] = ['GPE', 'ORG', 'PER', 'NAN', 'MISC', '']
role_to_type['Defendant'] = ['ORG', 'PER', 'GPE', 'NAN', 'MISC', '']
role_to_type['Destination'] = ['GPE', 'LOC', 'FAC', 'NAN', 'MISC', '']
role_to_type['Entity'] = ['PER', 'ORG', 'GPE', 'NAN', 'MISC', '']
role_to_type['Giver'] = ['PER', 'ORG', 'GPE', 'NAN', 'MISC', '']
role_to_type['Instrument'] = ['WEA', 'VEH', 'NAN', 'MISC', '']
role_to_type['Org'] = ['ORG', 'PER', 'NAN', 'MISC', '']
role_to_type['Origin'] = ['GPE', 'LOC', 'FAC', 'NAN', 'MISC', '']
role_to_type['Person'] = ['PER', 'NAN', 'MISC', '']
role_to_type['Place'] = ['FAC', 'GPE', 'LOC', 'NAN', 'MISC', '']
role_to_type['Plaintiff'] = ['PER', 'ORG', 'GPE', 'NAN', 'MISC', '']
role_to_type['Prosecutor'] = ['PER', 'ORG', 'GPE', 'NAN', 'MISC', '']
role_to_type['Recipient'] = ['PER', 'GPE', 'ORG', 'NAN', 'MISC', '']
role_to_type['Seller'] = ['GPE', 'PER', 'ORG', 'NAN', 'MISC', '']
role_to_type['Target'] = ['PER', 'FAC', 'VEH', 'LOC', 'ORG', 'WEA', 'NAN', 'MISC', '']
role_to_type['Vehicle'] = ['VEH', 'NAN', 'MISC', '']
role_to_type['Victim'] = ['PER', 'NAN', 'MISC', '']

verb_keywords = dict()

verb_keywords['Business:Declare-Bankruptcy'] = ['bankrupt']
verb_keywords['Business:End-Org'] = ['crumble', 'cease', 'close']
verb_keywords['Business:Merge-Org'] = ['merge']
verb_keywords['Business:Start-Org'] = ['found', 'create']
verb_keywords['Conflict:Attack'] = ['attack', 'fight']
verb_keywords['Conflict:Demonstrate'] = ['demonstrate']
verb_keywords['Contact:Meet'] = ['meet']
verb_keywords['Contact:Phone-Write'] = ['call', 'write']
verb_keywords['Justice:Acquit'] = ['acquit']
verb_keywords['Justice:Appeal'] = ['appeal']
verb_keywords['Justice:Arrest-Jail'] = ['arrest', 'capture', 'jail']
verb_keywords['Justice:Charge-Indict'] = ['charge', 'accuse', 'indict']
verb_keywords['Justice:Convict'] = ['convict']
verb_keywords['Justice:Execute'] = ['execute', 'hang']
verb_keywords['Justice:Extradite'] = ['extradite']
verb_keywords['Justice:Fine'] = ['fine']
verb_keywords['Justice:Pardon'] = ['pardon']
verb_keywords['Justice:Release-Parole'] = ['release', 'free']
verb_keywords['Justice:Sentence'] = ['sentence']
verb_keywords['Justice:Sue'] = ['sue']
verb_keywords['Justice:Trial-Hearing'] = ['hear']
verb_keywords['Life:Be-Born'] = []
verb_keywords['Life:Die'] = ['kill', 'die', 'suicide']
verb_keywords['Life:Divorce'] = ['divorce']
verb_keywords['Life:Injure'] = ['injure', 'wound', 'hurt']
verb_keywords['Life:Marry'] = ['marry', 'wed']
verb_keywords['Movement:Transport'] = ['go', 'come', 'arrive', 'move']
verb_keywords['Personnel:Elect'] = ['elect', 'vote']
verb_keywords['Personnel:End-Position'] = ['resign', 'retire']
verb_keywords['Personnel:Nominate'] = ['name', 'nominate']
verb_keywords['Personnel:Start-Position'] = ['appoint', 'hire']
verb_keywords['Transaction:Transfer-Money'] = ['pay']
verb_keywords['Transaction:Transfer-Ownership'] = ['buy', 'sell', 'seize']

noun_keywords = dict()

noun_keywords['Business:Declare-Bankruptcy'] = ['bankruptcy']
noun_keywords['Business:End-Org'] = []
noun_keywords['Business:Merge-Org'] = ['merger']
noun_keywords['Business:Start-Org'] = ['creation']
noun_keywords['Conflict:Attack'] = ['war', 'attack', 'fight']
noun_keywords['Conflict:Demonstrate'] = ['rally', 'protest', 'demonstration']
noun_keywords['Contact:Meet'] = ['meeting', 'meet', 'summit']
noun_keywords['Contact:Phone-Write'] = ['letters', 'call']
noun_keywords['Justice:Acquit'] = ['acquittal']
noun_keywords['Justice:Appeal'] = ['appeal']
noun_keywords['Justice:Arrest-Jail'] = ['arrest', 'capture']
noun_keywords['Justice:Charge-Indict'] = ['charge', 'indictment']
noun_keywords['Justice:Convict'] = ['conviction']
noun_keywords['Justice:Execute'] = ['execution']
noun_keywords['Justice:Extradite'] = ['extradition']
noun_keywords['Justice:Fine'] = ['fine']
noun_keywords['Justice:Pardon'] = ['pardon']
noun_keywords['Justice:Release-Parole'] = ['release']
noun_keywords['Justice:Sentence'] = []
noun_keywords['Justice:Sue'] = ['lawsuit']
noun_keywords['Justice:Trial-Hearing'] = ['trail', 'hearing']
noun_keywords['Life:Be-Born'] = ['birth', 'childbirth']
noun_keywords['Life:Die'] = ['death', 'dead']
noun_keywords['Life:Divorce'] = ['divorce']
noun_keywords['Life:Injure'] = ['injure', 'hurt', 'injury']
noun_keywords['Life:Marry'] = ['wedding', 'marriage']
noun_keywords['Movement:Transport'] = ['trip', 'visit', 'journey']
noun_keywords['Personnel:Elect'] = ['election']
noun_keywords['Personnel:End-Position'] = ['resignation', 'retirement']
noun_keywords['Personnel:Nominate'] = ['nomination']
noun_keywords['Personnel:Start-Position'] = ['appointment']
noun_keywords['Transaction:Transfer-Money'] = ['loan', 'donation']
noun_keywords['Transaction:Transfer-Ownership'] = ['acquisition']

verb_keywords_short = dict()

verb_keywords_short['Business:Declare-Bankruptcy'] = ['bankrupt']
verb_keywords_short['Business:End-Org'] = ['cease', 'close']
verb_keywords_short['Business:Merge-Org'] = ['merge']
verb_keywords_short['Business:Start-Org'] = ['create']
verb_keywords_short['Conflict:Attack'] = ['attack']
verb_keywords_short['Conflict:Demonstrate'] = ['demonstrate']
verb_keywords_short['Contact:Meet'] = ['meet']
verb_keywords_short['Contact:Phone-Write'] = ['write']
verb_keywords_short['Justice:Acquit'] = ['acquit']
verb_keywords_short['Justice:Appeal'] = ['appeal']
verb_keywords_short['Justice:Arrest-Jail'] = ['arrest', 'jail']
verb_keywords_short['Justice:Charge-Indict'] = ['charge', 'indict']
verb_keywords_short['Justice:Convict'] = ['convict']
verb_keywords_short['Justice:Execute'] = ['execute']
verb_keywords_short['Justice:Extradite'] = ['extradite']
verb_keywords_short['Justice:Fine'] = ['fine']
verb_keywords_short['Justice:Pardon'] = ['pardon']
verb_keywords_short['Justice:Release-Parole'] = ['release']
verb_keywords_short['Justice:Sentence'] = ['sentence']
verb_keywords_short['Justice:Sue'] = ['sue']
verb_keywords_short['Justice:Trial-Hearing'] = ['hear']
verb_keywords_short['Life:Be-Born'] = []
verb_keywords_short['Life:Die'] = ['die']
verb_keywords_short['Life:Divorce'] = ['divorce']
verb_keywords_short['Life:Injure'] = ['injure']
verb_keywords_short['Life:Marry'] = ['marry']
verb_keywords_short['Movement:Transport'] = ['go']
verb_keywords_short['Personnel:Elect'] = ['elect']
verb_keywords_short['Personnel:End-Position'] = ['retire']
verb_keywords_short['Personnel:Nominate'] = ['nominate']
verb_keywords_short['Personnel:Start-Position'] = ['hire']
verb_keywords_short['Transaction:Transfer-Money'] = ['pay']
verb_keywords_short['Transaction:Transfer-Ownership'] = ['buy']

noun_keywords_short = dict()

noun_keywords_short['Business:Declare-Bankruptcy'] = ['bankruptcy']
noun_keywords_short['Business:End-Org'] = []
noun_keywords_short['Business:Merge-Org'] = []
noun_keywords_short['Business:Start-Org'] = []
noun_keywords_short['Conflict:Attack'] = ['attack']
noun_keywords_short['Conflict:Demonstrate'] = ['demonstration']
noun_keywords_short['Contact:Meet'] = ['meet']
noun_keywords_short['Contact:Phone-Write'] = []
noun_keywords_short['Justice:Acquit'] = ['acquittal']
noun_keywords_short['Justice:Appeal'] = ['appeal']
noun_keywords_short['Justice:Arrest-Jail'] = ['arrest']
noun_keywords_short['Justice:Charge-Indict'] = ['charge', 'indictment']
noun_keywords_short['Justice:Convict'] = ['conviction']
noun_keywords_short['Justice:Execute'] = ['execution']
noun_keywords_short['Justice:Extradite'] = ['extradition']
noun_keywords_short['Justice:Fine'] = ['fine']
noun_keywords_short['Justice:Pardon'] = ['pardon']
noun_keywords_short['Justice:Release-Parole'] = ['release']
noun_keywords_short['Justice:Sentence'] = []
noun_keywords_short['Justice:Sue'] = []
noun_keywords_short['Justice:Trial-Hearing'] = ['trail', 'hearing']
noun_keywords_short['Life:Be-Born'] = ['birth', 'childbirth']
noun_keywords_short['Life:Die'] = ['death']
noun_keywords_short['Life:Divorce'] = ['divorce']
noun_keywords_short['Life:Injure'] = ['injury']
noun_keywords_short['Life:Marry'] = ['marriage']
noun_keywords_short['Movement:Transport'] = []
noun_keywords_short['Personnel:Elect'] = ['election']
noun_keywords_short['Personnel:End-Position'] = ['retirement']
noun_keywords_short['Personnel:Nominate'] = ['nomination']
noun_keywords_short['Personnel:Start-Position'] = ['appointment']
noun_keywords_short['Transaction:Transfer-Money'] = ['donation']
noun_keywords_short['Transaction:Transfer-Ownership'] = []

role_keywords = dict()

role_keywords['Adjudicator'] = ['adjudicator']
role_keywords['Agent'] = ['agent']
role_keywords['Artifact'] = ['artifact']
role_keywords['Attacker'] = ['attacker']
role_keywords['Beneficiary'] = ['beneficiary']
role_keywords['Buyer'] = ['buyer']
role_keywords['Defendant'] = ['defendant']
role_keywords['Destination'] = ['destination']
role_keywords['Entity'] = ['entity']
role_keywords['Giver'] = ['giver']
role_keywords['Instrument'] = ['instrument']
role_keywords['Org'] = ['organization']
role_keywords['Origin'] = ['origin']
role_keywords['Person'] = ['person']
role_keywords['Place'] = ['place']
role_keywords['Plaintiff'] = ['plaintiff']
role_keywords['Prosecutor'] = ['prosecutor']
role_keywords['Recipient'] = ['recipient']
role_keywords['Seller'] = ['seller']
role_keywords['Target'] = ['target']
role_keywords['Vehicle'] = ['vehicle']
role_keywords['Victim'] = ['victim']


class gurobi_opt:
    def __init__(self, predicate_score, argument_scores, entity_types, selected_event_types, selected_role_types,
                 weight=10,
                 prediction_length=5):
        self.num_predicate_labels = len(predicate_score)
        self.num_argument_labels = len(argument_scores[0])
        self.num_max = max(self.num_predicate_labels, self.num_argument_labels)
        self.initial_predicate_score = predicate_score + [0] * (self.num_max - self.num_predicate_labels)  # 1*num_max
        # self.initial_predicate_score = list(softmax(np.asarray(predicate_score + [0] * (self.num_max - self.num_predicate_labels))))
        self.initial_argument_scores = list()
        for tmp_arg_pos in range(len(argument_scores)):
            self.initial_argument_scores.append(
                argument_scores[tmp_arg_pos] + [0] * (self.num_max - self.num_argument_labels))  # n*num_max
            # self.initial_argument_scores.append(list(softmax(np.asarray(argument_scores[tmp_arg_pos] + [0] * (self.num_max - self.num_argument_labels)))))
        self.prediction_length = prediction_length
        self.selected_event_types = selected_event_types
        self.selected_role_types = selected_role_types
        self.entity_types = entity_types
        self.weight = weight

    def optimize_all(self):
        predicate_prediction = list()
        argument_predictions = list()
        for _ in range(len(self.initial_argument_scores)):
            argument_predictions.append(list())
        # make prediction for predicates
        for prediction_iteration in range(self.prediction_length):
            # print('We need to update the scores')
            tmp_predicate_score = list()
            for e_type_id, tmp_score in enumerate(self.initial_predicate_score):
                if e_type_id < self.num_predicate_labels and self.selected_event_types[
                    e_type_id] in predicate_prediction:
                    tmp_predicate_score.append(0)
                else:
                    tmp_predicate_score.append(tmp_score)
            input_scores = [np.asarray(tmp_predicate_score)]
            for tmp_arg_pos in range(len(self.initial_argument_scores)):
                tmp_argument_score = list()
                for r_type_id, tmp_score in enumerate(self.initial_argument_scores[tmp_arg_pos]):
                    tmp_argument_score.append(tmp_score)
                input_scores.append(np.asarray(tmp_argument_score))
            input_scores_array = np.asarray(input_scores)
            optimized_scores = self.optimize(input_scores_array)
            for tmp_pos in range(len(optimized_scores)):
                if tmp_pos == 0:
                    # predicate
                    for k in range(self.num_max):
                        if optimized_scores[tmp_pos][k] > 0:
                            if k > self.num_predicate_labels - 1:
                                predicate_prediction.append('None')
                            else:
                                predicate_prediction.append(self.selected_event_types[k])
                            break
                else:
                    continue
        # make prediction for arguments
        for prediction_iteration in range(self.prediction_length):
            # print('We need to update the scores')
            tmp_predicate_score = list()
            for e_type_id, tmp_score in enumerate(self.initial_predicate_score):
                tmp_predicate_score.append(tmp_score)
            input_scores = [np.asarray(tmp_predicate_score)]
            for tmp_arg_pos in range(len(self.initial_argument_scores)):
                tmp_argument_score = list()
                for r_type_id, tmp_score in enumerate(self.initial_argument_scores[tmp_arg_pos]):
                    if r_type_id < self.num_argument_labels and self.selected_role_types[r_type_id] in \
                            argument_predictions[tmp_arg_pos]:
                        tmp_argument_score.append(0)
                    else:
                        tmp_argument_score.append(tmp_score)
                input_scores.append(np.asarray(tmp_argument_score))
            input_scores_array = np.asarray(input_scores)
            optimized_scores = self.optimize(input_scores_array)
            for tmp_pos in range(len(optimized_scores)):
                if tmp_pos == 0:
                    # predicate
                    continue
                else:
                    # arguments
                    for k in range(self.num_max):
                        if optimized_scores[tmp_pos][k] > 0:
                            # argument_predictions[tmp_pos - 1].append(self.selected_role_types[k])
                            if k > self.num_argument_labels - 1:
                                argument_predictions[tmp_pos - 1].append('None')
                            else:
                                argument_predictions[tmp_pos - 1].append(self.selected_role_types[k])
                            break
        return predicate_prediction, argument_predictions

    def optimize(self, input_scores):
        self.model = Model('lp')
        self.model.setParam('OutputFlag', False)
        self.x = self.model.addVars(input_scores.shape[0], self.num_max, lb=0.0, ub=1.0, obj=input_scores,
                                    vtype=GRB.INTEGER,
                                    name="x")
        # For each prediction, we can only predict one result
        self.model.addConstrs((self.sum_prob(i) == 1.0 for i in range(input_scores.shape[0])), name='prob_constrs')
        # For each event, we cannot assign multiple arguments to the same role
        self.model.addConstrs((self.pred_prob(input_scores.shape[0] - 1, k) <= 1.0 for k in range(self.num_max)),
                              name='no_repeat_prediction')
        # Add constraints from the definitions
        for tmp_e_position, tmp_e_type in enumerate(self.selected_event_types):
            for tmp_r_position, tmp_r_type in enumerate(self.selected_role_types):
                if tmp_r_type not in event_schema_to_roles[tmp_e_type]:
                    self.model.addConstrs(
                        (self.x[0, tmp_e_position] + self.x[tmp_row + 1, tmp_r_position] <= 1.0 for tmp_row in
                         range(input_scores.shape[0] - 1)), name='event_definition')
        #
        # add constraints from the entity types
        for tmp_argument_pos in range(input_scores.shape[0]-1):
            for tmp_r_position, tmp_r_type in enumerate(self.selected_role_types):
                if self.entity_types[tmp_argument_pos] not in role_to_type[tmp_r_type]:
                    self.model.addConstr(self.x[tmp_argument_pos + 1, tmp_r_position] == 0.0)

        for tmp_e_position, tmp_e_type in enumerate(self.selected_event_types):
            possible_entitie_types = list()
            for tmp_r in event_schema_to_roles[tmp_e_type]:
                possible_entitie_types += role_to_type[tmp_r]
            for tmp_entity_type in self.entity_types:
                if tmp_entity_type not in possible_entitie_types:
                    self.model.addConstr(self.x[0, tmp_e_position] == 0.0)
                    break

        self.model.update()
        self.sum_score = 0.0
        for i in range(input_scores.shape[0]):
            if i == 0:
                for k in range(self.num_max):
                    self.sum_score += self.x[i, k] * input_scores[i][k] * self.weight * (input_scores.shape[0] - 1)
                    # self.sum_score += self.x[i, k] * input_scores[i][k] * 99 * (input_scores.shape[0] - 1)
            else:
                for k in range(self.num_max):
                    self.sum_score += self.x[i, k] * input_scores[i][k]
        self.model.setObjective(self.sum_score, GRB.MAXIMIZE)  # maximize profit
        self.model.optimize()
        # print(input_scores)
        results = list()
        for i in range(input_scores.shape[0]):
            tmp_scores = list()
            for k in range(self.num_max):
                tmp_scores.append(self.x[i, k].X)
            results.append(tmp_scores)
        # print(results)
        return results

    # self.model.printAttr('X')

    def __call__(self):
        for v in self.model.getVars():
            print('%s %g' % (v.varName, v.x))
        return self.model.getVars()

    def sum_prob(self, i):
        sum_prob = 0.0
        for k in range(self.num_max):
            sum_prob += self.x[i, k]
        return sum_prob

    def pred_prob(self, num_arguments, k):
        sum_prob = 0.0
        for arg_id in range(num_arguments):
            sum_prob += self.x[arg_id + 1, k]
        return sum_prob


def get_represetation(sentence, target_positions, tokenizer, model, device, representation_type='all'):
    start_position = target_positions[0]
    end_position = target_positions[1]
    tokens = list()
    masks = list()
    # new_tokens = tokenizer.encode('<s>', add_special_tokens=False)
    new_tokens = tokenizer.encode('[CLS]', add_special_tokens=False)
    tokens += new_tokens
    masks += [1] * len(new_tokens)
    token_start_position = 0
    token_end_position = -1

    for i, w in enumerate(sentence):
        if i == start_position:
            token_start_position = len(tokens)
        if i == end_position:
            token_end_position = len(tokens)
        if start_position <= i < end_position:
            if representation_type == 'all':
                new_tokens = tokenizer.encode(w, add_special_tokens=False)
            else:
                new_tokens = tokenizer.encode(tokenizer.mask_token, add_special_tokens=False)
            tokens += new_tokens
            masks += [0] * len(new_tokens)
        else:
            new_tokens = tokenizer.encode(w, add_special_tokens=False)
            tokens += new_tokens
            masks += [1] * len(new_tokens)
    # new_tokens = tokenizer.encode('</s>', add_special_tokens=False)
    new_tokens = tokenizer.encode('[SEP]', add_special_tokens=False)
    tokens += new_tokens
    masks += [1] * len(new_tokens)
    if len(tokens) > 512:
        return 'Too long'
    tensorized_token = torch.tensor([tokens]).to(device)
    tensorized_mask = torch.tensor([masks]).to(device)
    if representation_type == 'all':
        resulted_embedding = torch.mean(model(tensorized_token)[0][:, token_start_position:token_end_position, :],
                                        dim=1)
    else:
        resulted_embedding = torch.mean(
            model(tensorized_token, attention_mask=tensorized_mask)[0][:, token_start_position:token_end_position, :],
            dim=1)
    return torch.tensor(resulted_embedding.tolist()).view(1, -1).to(device)
