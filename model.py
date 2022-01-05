import torch 
import csv
import os
import json
import tqdm
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torchinfo import summary
from torch.autograd import Variable
from torch import optim
from utils import get_attn_pad_mask, get_attn_subsequence_mask, get_item_pad_mask
from load_config import get_attribute
from txt_encoder import txtEncoder

#忽略警告
import warnings
warnings.filterwarnings("ignore")



class PosEmbedding(nn.Module):		#位置embedding策略可修改
	"""docstring for PosEmbedding"""
	def __init__(self, d_model, dropout=0.2, max_length=100):
		super(PosEmbedding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		pe = torch.zeros(max_length, d_model)
		position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
		# div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		div_term = torch.arange(0, d_model, 2).float() * (-math.log(100.0) / d_model)

		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)
	
	def forward(self, x):
		x = x + self.pe[:x.size(0), :]  #将word embedding与positional embedding相加
		return self.dropout(x)


		
class Item2CPEncoder(nn.Module):
	def __init__(self, item_n, d_k, d_v, n_heads, d_model, d_ff, n_layers):
		super(Item2CPEncoder, self).__init__()
		self.d_model = d_model
		self.item_emb = nn.Embedding(item_n, d_model)
		self.linear1 = nn.Linear(d_model, d_model)
		self.vec_q = nn.Linear(d_model, 1)
		self.softmax = nn.Softmax(dim=0)

	def get_items_sum(self, items):
		alpha = self.vec_q(torch.sigmoid(self.linear1(items)))
		alpha = self.softmax(alpha)
		res = torch.sum(alpha * items, dim=0)
		return res


	def forward(self, inputs, length_data):
		inputs_emb = self.item_emb(inputs)

		outputs = inputs_emb

		for user_idx, output in enumerate(outputs):
			if user_idx >= len(length_data):
				break
			lengths = length_data[user_idx]
			basket_outputs = []
			for basket_idx, basket in enumerate(output):
				if basket_idx >= len(lengths):
					break
				length = lengths[basket_idx]

				items = basket[:length]
				pitems = self.get_items_sum(items)

				basket_outputs.append(pitems)
			basket_outputs = torch.stack(basket_outputs)	#size:[day, emb_size]

			ans_outputs.append(basket_outputs)

		return ans_outputs	#size:[batch, day, emb_size]
		

class CPEncoder(nn.Module):
	"""docstring for Encoder"""
	def __init__(self, d_model, d_k, d_v, n_heads, d_ff, n_layers):
		super(CPEncoder, self).__init__()
		self.pos_emb = PosEmbedding(d_model)
		self.sub_layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads) for _ in range(n_layers)])
		self.attn_layer = nn.Linear(d_model*2, d_model)

	def get_pad_attn_mask(self, inputs_q, inputs_k):

		batch_size, len_q = inputs_q.shape
		batch_size, len_k = inputs_k.shape

		pad_attn_mask = inputs_k.data.eq(0)
		return pad_attn_mask

	def forward(self, inputs, inputs_emb):
		
		outputs = self.pos_emb(inputs_emb.transpose(0,1)).transpose(0,1).cuda()
		pad_attn_mask = self.get_pad_attn_mask(inputs, inputs)

		pad_attn_mask = pad_attn_mask.cuda()

		outputs = outputs.transpose(0,1)
		for layer in self.sub_layers:
			outputs = layer(outputs, src_key_padding_mask=pad_attn_mask)

		queries = outputs.transpose(0,1)
		lengths = []
		final_outputs = []


		for length in inputs:	#获取每个user实际有效的basket个数
			lengths.append(int(sum(length)))

		#attention
		softmax_res_list = []
		queries = queries.cuda()
		for idx, query in enumerate(queries):
			cur_query = query[-1]
			repeat_cur_query = cur_query.repeat(lengths[idx],1)


			cat_basket = torch.cat((repeat_cur_query, attn_values), 1)
			attn_basket = self.attn_layer(cat_basket)
			sum_attn_basket = attn_basket.sum(dim=-1)

			softmax_res = F.softmax(sum_attn_basket, dim=0).view(1,-1)

			final_res = torch.mm(softmax_res, attn_values)
			final_outputs.append(final_res)

		final_outputs = torch.stack(final_outputs)
		return final_outputs


class Generator(nn.Module):
	"""docstring for Generator"""
	def __init__(self, input_dim, gen_hidden1, gen_hidden2, gen_hidden3, output_dim):
		super(Generator, self).__init__()
		self.gen = nn.Sequential(
			nn.Linear(input_dim, gen_hidden1),
			nn.ReLU(),
			nn.Linear(gen_hidden1, output_dim),
			nn.Tanh()
		)
	def forward(self, x):
		x = self.gen(x)
		return x
		


class Discriminator(nn.Module):
	"""docstring for Discriminator"""
	def __init__(self, input_dim, dis_hidden1, dis_hidden2):
		super(Discriminator, self).__init__()
		self.dis = nn.Sequential(
			nn.Linear(input_dim, dis_hidden1),
			nn.LeakyReLU(0.2),
			nn.Linear(dis_hidden1, 1),
			nn.Sigmoid()
		)
	def forward(self, x):
		x = self.dis(x)
		return x
		
		


class CPDecoder(nn.Module):
	"""docstring for CPDecoder"""
	def __init__(self, d_model, d_output, output_dim, dropout, vocab_length):
		super(CPDecoder, self).__init__()
		self.d_model = d_model
		self.d_output = d_output
		self.output_dim = output_dim

		self.Itemmodel = Item2CPEncoder(get_attribute('items_total'),
                       get_attribute('d_k'),
                       get_attribute('d_v'),
                       get_attribute('n_heads'),
                       get_attribute('d_model'),
                       get_attribute('d_ff'),
                       get_attribute('n_layers')
    		)

		self.CPmodel = CPEncoder(get_attribute("d_model"),
                    get_attribute("d_k"),
                    get_attribute("d_v"),
                    get_attribute("n_heads"),
                    get_attribute("d_ff"),
                    get_attribute("n_layers"),
    		)


		self.txt_model = txtEncoder(vocab_length, 
					get_attribute("txt_d_model"),
					get_attribute("txt_n_heads"),
					get_attribute("txt_d_k"),
					get_attribute("txt_d_v"),
					get_attribute("txt_n_layers"),
					get_attribute("txt_d_ff"),
					get_attribute("d_model")
			)

		self.dropout = nn.Dropout(p=dropout)
		self.W_d = nn.Linear(d_model, d_output)
		self.W_d_output = nn.Sequential(nn.Linear(d_output, output_dim))
		self.W_alpha = nn.Sequential(nn.Linear(output_dim, output_dim), nn.Sigmoid())
		self.W_txt = nn.Sequential(nn.Linear(d_model, output_dim), nn.Sigmoid())


	def forward(self, train_data, length_data, user_txts, user_txt_masks, user_data, frequency_data):

		item2cp_output = self.Itemmodel(train_data, length_data)

		baskets_emb = rnn_utils.pad_sequence(item2cp_output, batch_first=True, padding_value=0) 

		cp_enc_inputs = []
		for user_idx, baskets in enumerate(item2cp_output):
			length = baskets.shape[0]
			max_length = baskets_emb.shape[1]
			cp_enc_input = [1 for _ in range(length)]
			cp_enc_input.extend([0 for _ in range(max_length - length)])
			cp_enc_inputs.append(torch.tensor(cp_enc_input))
		caonima = cp_enc_inputs
		cp_enc_inputs = torch.stack(cp_enc_inputs)

		cp_outputs = self.CPmodel(cp_enc_inputs, baskets_emb)

		cp_outputs = self.W_d(cp_outputs.squeeze(1))

		cp_outputs = self.W_d_output(cp_outputs)

		txt_output, _ = self.txt_model(user_txts, user_txt_masks)



		user_frequency = train_data[0].new_zeros(len(train_data), self.output_dim, dtype=torch.float)
		for idx, user in enumerate(user_data):
			item_freq_dic = frequency_data[user]
			for key, value in item_freq_dic.items():
				user_frequency[idx, key] = value

		l_p = user_frequency
		beta = l_p.clone()
		beta[beta != 0] = 1	#l_p的one-hot向量
		alpha = self.W_alpha(l_p)

		output = (1 - alpha * beta) * cp_outputs + (alpha * l_p) + 0.5*self.W_txt(txt_output)
		
		return output, cp_outputs	

		
		






























