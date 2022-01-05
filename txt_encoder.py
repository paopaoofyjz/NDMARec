import math
import json
import csv
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.utils.data as Data
from torchinfo import summary
from load_config import get_attribute


class PosEmbedding(nn.Module):
	"""docstring for PosEmbedding"""
	def __init__(self, d_model, dropout=0.1, max_length=500):
		super(PosEmbedding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		pe = torch.zeros(max_length, d_model)
		position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)
	
	def forward(self, x):
		x = x + self.pe[:x.size(0), :]  #将word embedding与positional embedding相加
		return self.dropout(x)


class MutiHeadAttention(nn.Module):
	"""docstring for MutiHeadAttention"""
	def __init__(self, d_model, n_heads, d_k, d_v):
		super(MutiHeadAttention, self).__init__()
		self.d_model = d_model
		self.d_k = d_k
		self.d_v = d_v
		self.n_heads = n_heads
		self.W_Q = nn.Linear(d_model, d_k*n_heads, bias=True)			#用于生成q注意向量
		self.W_K = nn.Linear(d_model, d_k*n_heads, bias=True)			#用于生成k注意向量
		self.W_V = nn.Linear(d_model, d_v*n_heads, bias=True)			#用于生成v注意向量
		self.trans_layer = nn.Linear(d_v*n_heads, d_model, bias=True)	#把向量维度转成d_model,方便后续连接

	def forward(self, input_Q, input_K, input_V, attn_mask):			#input为输入的embedding,attn_mask控制注意力集中在有效单词的embedding上，对用于补齐长度的无效单词的embedding的注意力权重置为0
		residual, batch_size = input_Q, input_Q.shape[0]				#residual为残差连接的原始值

		Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)		#batch_size*n_heads*len_q*d_k
		K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
		V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)

		attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)			
		Q_K_score = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)		#batch_size*n_heads*len_q*len_k
		Q_K_score.masked_fill_(attn_mask, -1e9)
		attn_ratio = nn.Softmax(dim=-1)(Q_K_score)
		self_attn = torch.matmul(attn_ratio, V).transpose(1,2).reshape(batch_size, -1, self.n_heads*self.d_v)
		attn_output = self.trans_layer(self_attn)
		return nn.LayerNorm(self.d_model).cuda()(attn_output + residual), self_attn

	
class EncoderLayer(nn.Module):
	"""docstring for EncoderLayer"""
	def __init__(self, d_model, n_heads, d_k, d_v, d_ff):
		super(EncoderLayer, self).__init__()
		self.d_model = d_model
		self.attn = MutiHeadAttention(d_model, n_heads, d_k, d_v)
		self.ffn1 = nn.Linear(d_model, d_ff, bias=True)
		self.ReLU = nn.ReLU()
		self.ffn2 = nn.Linear(d_ff, d_model, bias=True)
	
	def forward(self, inputs_emb, pad_attn_mask):
		attn_outputs, attn_ratio = self.attn(inputs_emb, inputs_emb, inputs_emb, pad_attn_mask)
		outputs = self.ffn1(attn_outputs)
		outputs = self.ReLU(outputs)
		outputs = self.ffn2(outputs)
		return nn.LayerNorm(self.d_model).cuda()(inputs_emb + outputs), attn_ratio		#残差连接 and 注意力权重
		
		

class txtEncoder(nn.Module):
	"""docstring for Encoder"""
	def __init__(self, vocab_size, d_model, n_heads, d_k, d_v, n_layers, d_ff, cp_d_model):
		super(txtEncoder, self).__init__()
		self.word_emb = nn.Embedding(vocab_size, d_model)
		self.pos_emb = PosEmbedding(d_model)
		self.sub_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_k, d_v, d_ff) for _ in range(n_layers)])
		self.linear = nn.Linear(d_model, d_model)
		self.soft_attn = nn.Linear(d_model, 1)
		self.output_layer = nn.Linear(d_model, cp_d_model)

	def get_pad_attn_mask(self, inputs_q, inputs_k):
		# print(inputs.shape)
		batch_size, len_q = inputs_q.shape
		batch_size, len_k = inputs_k.shape

		pad_attn_mask = inputs_k.data.eq(0).unsqueeze(1)  
		return pad_attn_mask.expand(batch_size, len_q, len_k)		#batch_size*len_q*len_k

	def forward(self, enc_inputs, enc_inputs_mask):
		# enc_inputs = enc_inputs.cuda()			#batch_size*sentence_len
		inputs_emb = self.word_emb(enc_inputs)	#batch_size*sentence_len*d_model
		# print("inputs_emb size:", inputs_emb.size())
		outputs = self.pos_emb(inputs_emb.transpose(0,1)).transpose(0,1)
		pad_attn_mask = self.get_pad_attn_mask(enc_inputs, enc_inputs)
		enc_self_attns = []
		for layer in self.sub_layers:
			outputs, attn = layer(outputs, pad_attn_mask)
			enc_self_attns.append(attn)

		linear_output = self.linear(outputs)
		alpha = self.soft_attn(torch.sigmoid(linear_output))
		res = alpha * enc_inputs_mask * outputs 
		res_sum = torch.sum(res, dim=1)

		res_sum = self.output_layer(res_sum)

		return res_sum, enc_self_attns

def get_data():
	diag_info_dic = {}
	with open("./data/diag_info_divwords.json", 'r', encoding='utf-8') as f:
		diag_info_dic = json.load(f)
	
	sentences = {}
	words_size = []

	vocab = []
	with open("./data/vocab.txt", 'r', encoding='utf-8') as f:
		vocab = f.read()

	vocab = eval(vocab)		#词库（词袋）
	vocab_dic = {w:i for i,w in enumerate(vocab)}

	for no in diag_info_dic:
		if len(diag_info_dic[no]) > 4:
			vocab.extend(diag_info_dic[no])
			words_size.append(len(diag_info_dic[no]))
			sentences[no] = diag_info_dic[no]

	max_length = max(words_size)+1
	
	for key in sentences:	#把所有句子补成等长
		if len(sentences[key]) <= max_length:
			sentences[key].extend(["P" for i in range(max_length - len(sentences[key]))])
	return sentences, vocab_dic

def makedata(sentences, vocab_dic):
	enc_inputs = {}
	for key in sentences:
		enc_input = [[vocab_dic[w] for w in sentences[key]]]
		enc_inputs[key] = enc_input
		# enc_inputs.extend(enc_input)
	# return torch.LongTensor(enc_inputs)
	return enc_inputs
	

def return_txt_data():
	sentences, vocab_dic = get_data()
	enc_inputs = makedata(sentences, vocab_dic)

	return enc_inputs, len(vocab_dic)




















