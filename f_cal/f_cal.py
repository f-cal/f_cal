import torch
import torch.nn as nn

import random
import numpy as np

class f_Cal(nn.Module):
	def __init__(self, K=100, divergence="kld", num_samples=200, seed=42, device='cuda'):
		super(f_Cal, self).__init__()
		self.K = K 
		self.divergence = divergence
		self.num_samples = num_samples
		self.device = device

		random.seed(seed)

	def forward(self, y, mu, std):
		chi_sq_samples = self.get_chi_sq_samples(y, mu, std)
		f_cal_loss = self.compute_divergence(chi_sq_samples)

		return f_cal_loss

	def get_chi_sq_samples(self, y, mu, std):
		total_samples = y.numel()
		chi_sq_samples = []
		for i in range(self.num_samples):
			idx = random.sample(range(0, total_samples), self.K)
			sample_y, sample_mu, sample_std = y[idx], mu[idx], std[idx]
			sample_q = (((sample_y - sample_mu)/sample_std)**2).sum()
			chi_sq_samples.append(sample_q)

		return torch.stack(chi_sq_samples)

	def compute_divergence(self, chi_sq_samples):
		# Computing Mean and Variance of the Empirical Chi-Squared Distribution
		emp_mu, emp_var = chi_sq_samples.mean(), chi_sq_samples.var()
		# Estimating Divergence
		if self.divergence == "kld":
			pred_dist = torch.distributions.normal.Normal(emp_mu, torch.sqrt(emp_var))
			true_dist = torch.distributions.normal.Normal(self.K*torch.ones([1]).to(self.device), np.sqrt(2*self.K)*torch.ones([1]).to(self.device))
			f_cal_loss = torch.distributions.kl.kl_divergence(pred_dist, true_dist).mean()
		elif self.divergence == "w-dist":
			f_cal_loss = ((emp_mu - self.K)**2 + emp_var + 2*K - 2*(emp_var*2*self.K)**0.5)

		return f_cal_loss 
