import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from model import Item2CPEncoder, CPEncoder, Generator, Discriminator, CPDecoder
from txt_encoder import return_txt_data
from utils import get_txt_mask, save_model, WeightMSELoss
from load_config import get_attribute
from metric import get_metric
from data import get_data_loader
from model_test import Test
import pandas as pd
import numpy as np
import random
import os
from tqdm import tqdm
from torchinfo import summary

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


data_path = "./data/length7+_cleaned_cp_data_train_valid_test.json"
model_folder = "./saves/"

# print("items_total:", get_attribute("items_total"))

def create_optimizer(model, epoch_controller_idx):
	optimizer = torch.optim.Adam(model.parameters(), 
								 lr=get_attribute("learning_rate")[epoch_controller_idx],
								 weight_decay=get_attribute("weight_decay")[epoch_controller_idx]
								 )
	return optimizer


def train_test():
	G = Generator(get_attribute("items_total"),
              get_attribute("gen_hidden1"),
              get_attribute("gen_hidden2"),
              get_attribute("gen_hidden3"),
              get_attribute("items_total")
    	).cuda()

	D = Discriminator(get_attribute("items_total"),
	                  get_attribute("dis_hidden1"),
	                  get_attribute("dis_hidden2")
	    ).cuda()


	txt_data, vocab_length = return_txt_data()
	model = CPDecoder(get_attribute("d_model"),
					  get_attribute("d_output"),
	                  get_attribute("items_total"),
	                  get_attribute("dropout"),
	                  vocab_length
	    ).cuda()
	model.load_state_dict(torch.load("./saves/12.21-model_epoch_298.pkl"))	#莫名终止了，所以从上一次保存的模型重新开始
	model.train()
	summary(model)

	criterion = WeightMSELoss()
	adv_criterion = nn.BCELoss()
	d_optimizer = torch.optim.Adam(D.parameters(), lr=0.00003)
	g_optimizer = torch.optim.Adam(G.parameters(), lr=0.00003)
	rec_optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
	txt_data, vocab_length = return_txt_data()
	epoch_controller_idx = 0	#超参数变动控制

	if get_attribute("test") == 1:
		Test(model, data_path)
		exit(0)


	for epoch in range(0, get_attribute("epochs")):

		train_data_loader = get_data_loader(data_path, 'train', get_attribute("batch_size"), False)
		train_tqdm = tqdm(train_data_loader)

		train_rec_loss = 0.0
		total_g_loss = 0.0
		total_d_loss = 0.0
		for train_idx, (user_data, train_data, truth_data, length_data, frequency_data) in enumerate(train_tqdm):

			train_data = rnn_utils.pad_sequence(train_data, batch_first=True, padding_value=0)

			target_onehot = []
			for truth in truth_data:
				one_hot = torch.zeros(get_attribute('items_total'))
				for item in truth:
					one_hot[item] = 1
				target_onehot.append(torch.tensor(one_hot))
			target_onehot = torch.stack(target_onehot)
			y_target = target_onehot.clone()

			user_txts, user_txt_masks = get_txt_mask(txt_data, user_data)

			rec_optimizer.zero_grad()
			cp_rec_target = y_target
			train_data = train_data.cuda()

			user_txts = torch.tensor(user_txts).cuda()
			user_txt_masks = torch.tensor(user_txt_masks).cuda()
			cp_rec_output, rec_cp_outputs = model(train_data, length_data, user_txts, user_txt_masks, user_data, frequency_data)     #cp_rec_output为模型所预测的最终输出次日结果，y_target为实际结果

			cp_rec_target = cp_rec_target.cuda()
			loss = criterion(truth=cp_rec_target, predict=cp_rec_output)

			if epoch >= 0:
				loss.backward()
				rec_optimizer.step()
				train_rec_loss += loss

			else:
				adv_real_input = []
				for labels in target_onehot:
					for idx, _ in enumerate(labels):
						labels[idx] = labels[idx] + random.uniform(0, 0.1)
					adv_real_input.append(labels)
				adv_real_input = torch.stack(adv_real_input).cuda()    #真正的后续item，为了判别器以0-1简单区分真假，故对真正的label添加抖动
				adv_real_label = Variable(torch.ones(target_onehot.shape[0], 1))
				adv_fake_input = rec_cp_outputs
				adv_fake_label = Variable(torch.zeros(target_onehot.shape[0], 1))

		    	#训练判别器
				d_optimizer.zero_grad()     #判别器梯度置零
				real_out = D(adv_real_input)    #得到真实cp的判别结果，与真实label求loss
				real_out = real_out.squeeze(1).cuda()
				adv_real_label = adv_real_label.cuda()
				d_loss_real = adv_criterion(real_out, adv_real_label)   #求判别器对于真实cp的loss

				adv_fake_input = adv_fake_input.cuda()
				gen_adv_fake_output = G(adv_fake_input).detach()     #输入噪声得到生成器结果

				fake_out = D(gen_adv_fake_output)    #得到噪声cp的判别结果，与噪声label求loss
				adv_fake_label = adv_fake_label.cuda()
				d_loss_fake = adv_criterion(fake_out, adv_fake_label) #求判别器对于假的cp的与其假label的loss
		    
				d_loss = d_loss_real + d_loss_fake
				total_d_loss += d_loss
				d_loss.backward()
				d_optimizer.step()
		    
		    	#训练生成器
				g_optimizer.zero_grad()
				adv_fake_input2 = G(adv_fake_input)
				fake_out2 = D(adv_fake_input2)
				g_loss = criterion(adv_fake_input2, cp_rec_target)
				adv_loss = adv_criterion(fake_out2, adv_real_label)


				total_g_loss += adv_loss

				total_loss = loss + 0.5*g_loss

				train_rec_loss += loss
			
				adv_loss.backward(retain_graph=True)
				total_loss.backward()
				g_optimizer.step()
				rec_optimizer.step()

			if hasattr(torch.cuda, 'empty_cache'):
				torch.cuda.empty_cache()

			print("\ntrain epoch:",epoch, "rec train loss:", train_rec_loss.data/((train_idx+1)), "generator loss:", total_g_loss/(train_idx+1), "discriminator loss:", total_d_loss/(train_idx+1))

		print("*"*50)
		print("train epoch", epoch, "total train loss:", train_rec_loss)
		print("*"*50)

		valid_data_loader = get_data_loader(data_path, 'valid', 256, False)
		valid_tqdm = tqdm(valid_data_loader)
		test_data_loader = get_data_loader(data_path, 'test', 256, False)
		test_tqdm = tqdm(test_data_loader)

		with torch.no_grad():
            # 验证损失
			model.eval()
			valid_total_loss = 0.0
			valid_pred = []
			valid_true = []
			for valid_idx, (user_data, valid_data, truth_data, length_data, frequency_data) in enumerate(valid_tqdm):
				

				valid_data = rnn_utils.pad_sequence(valid_data, batch_first=True, padding_value=0)
				
				target_onehot = []
				for truth in truth_data:
					one_hot = torch.zeros(get_attribute('items_total'))
					for item in truth:
						one_hot[item] = 1
					target_onehot.append(torch.tensor(one_hot))
				target_onehot = torch.stack(target_onehot)
				y_target = target_onehot.clone()

				user_txts, user_txt_masks = get_txt_mask(txt_data, user_data)

				cp_rec_target = y_target

				valid_data = valid_data.cuda()
				user_txts = torch.tensor(user_txts).cuda()
				user_txt_masks = torch.tensor(user_txt_masks).cuda()

				cp_rec_output, rec_cp_outputs = model(valid_data, length_data, user_txts, user_txt_masks, user_data, frequency_data)     #cp_rec_output为模型所预测的最终输出次日结果，y_target为实际结果
				cp_rec_target = cp_rec_target.cuda()
				loss = criterion(truth=cp_rec_target, predict=cp_rec_output)
				valid_total_loss += loss

				valid_true.append(cp_rec_target)
				valid_pred.append(cp_rec_output)

				print("\nvalid epoch:",epoch, "rec valid loss:", valid_total_loss/((valid_idx+1)))

			print("*"*50)
			print("valid epoch", epoch, "total valid loss:", valid_total_loss)
			print("*"*50)

			test_total_loss = 0.0
			test_true = []
			test_pred = []
			for test_idx, (user_data, test_data, truth_data, length_data, frequency_data) in enumerate(test_tqdm):

				test_data = rnn_utils.pad_sequence(test_data, batch_first=True, padding_value=0)

				target_onehot = []
				for truth in truth_data:
					one_hot = torch.zeros(get_attribute('items_total'))
					for item in truth:
						one_hot[item] = 1
					target_onehot.append(torch.tensor(one_hot))
				target_onehot = torch.stack(target_onehot)
				y_target = target_onehot.clone()

				user_txts, user_txt_masks = get_txt_mask(txt_data, user_data)
			    
				cp_rec_target = y_target

				test_data = test_data.cuda()
				user_txts = torch.tensor(user_txts).cuda()
				user_txt_masks = torch.tensor(user_txt_masks).cuda()

				cp_rec_output, rec_cp_outputs = model(test_data, length_data, user_txts, user_txt_masks, user_data, frequency_data)     #cp_rec_output为模型所预测的最终输出次日结果，y_target为实际结果
				cp_rec_target = cp_rec_target.cuda()
				loss = criterion(truth=cp_rec_target, predict=cp_rec_output)
				test_total_loss += loss

				test_true.append(cp_rec_target)
				test_pred.append(cp_rec_output)

				print("\ntest epoch:",epoch, "rec test loss:", test_total_loss/((test_idx+1)))

			print("*"*50)
			print("test epoch", epoch, "total test loss:", test_total_loss)
			print("*"*50)

			print("Valid Metirc ...")	#验证集测试
			valid_true = torch.cat(valid_true, dim=0)
			valid_pred = torch.cat(valid_pred, dim=0)
			valid_scores = get_metric(valid_true, valid_pred)
			print(valid_scores)
			print("*"*50)

			print("Test Metirc ...")	#测试集测试
			test_pred = torch.cat(test_pred, dim=0)
			test_true = torch.cat(test_true, dim=0)
			test_scores = get_metric(test_true, test_pred)
			print(test_scores)
			print("*"*50)


		model_path = f"{model_folder}/model_ma(6-6)_epoch_{epoch}.pkl"
		save_model(model, model_path)
		print(f"model save as {model_path} with loss {valid_total_loss}")

		model.train()





def main():
	train_test()
	



if __name__ == '__main__':
	main()



