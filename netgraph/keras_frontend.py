from argparse import Namespace
from collections import defaultdict
import itertools as it


class NoParam:
	pass



def params(ps, name, skip=[]):
	skip, ps_dict = set(skip), dict(ps)
	
	def decorator(func):
		def wrapper(self, *args, **kwargs):

			for (key, _), arg in zip(ps, args):
				kwargs[key] = arg

			# get name
			if 'name' in kwargs:
				given_name = kwargs['name']
				del kwargs['name']
			else:
				self._names_cnt[name] += 1
				given_name = name + '_' + str(self._names_cnt[name])
			self._last_given_name = given_name

			# check correct params
			not_in_ps = set(kwargs) - set(ps_dict)
			if not not_in_ps.issubset(skip):
				raise ValueError(f"Unexpected parameters passed: {not_in_ps - skip}")

			# set default params
			for p in set(ps_dict) - set(kwargs):
				if ps_dict[p] == NoParam:
					raise ValueError(f'Parameter {p} not set')
				kwargs[p] = ps_dict[p]

			if self.verbose:
				print('Processing', func.__name__, 'as', given_name)

			return func(self, given_name, Namespace(**kwargs))
		return wrapper
	return decorator


import itertools as it

def str_product(xs, ys, delim=''):
	yield from it.starmap(lambda x, y: str(x) + delim + str(y),  it.product(xs, ys))

def common_skip(pxs):
	sxs  = [ 'initializer', 'regularizer', 'constraint' ]
	return list(str_product(pxs, sxs, '_'))



# TODO: probably should reimplement this as a metaclass in order to allow
#       'from <mode> import *' imports
class KerasFrontend:

	def __init__(self, ng):
		self.ng = ng
		self._names_cnt = defaultdict(int)
		self._last_given_name = None
		self.verbose = True


	@params([
		('filters', NoParam), ('kernel_size', NoParam),
		('strides', (1, 1)), ('padding', 'valid'), 
		('activation', None), ('use_bias',True), 
	], name='conv2d', skip = common_skip([
		'kernel', 'bias', 'activity'
	]))
	def Conv2D(self, name, pm):

		def func(input):
			
			assert pm.use_bias # TODO: use_bias == False

			res = self.ng.conv2d( (input, name + '/BiasAdd'), 
				(name + '/kernel', name + '/bias'), 
				padding=pm.padding, strides=pm.strides)

			if pm.activation == 'relu':
				res = self.ng.relu( (res, name + '/Relu') )

			else:
				assert pm.activation is None

			return res

		return func


	@params([
		('units', NoParam), 
		('activation', None), 
		('use_bias', True), 
	], name='dense', skip = common_skip([
		'kernel', 'bias', 'activity'
	]))
	def Dense(self, name, pm):
		def func(input):
			assert pm.use_bias # TODO: use_bias == False

			
			res = self.ng.dense( (input, name + '/BiasAdd'), (name + '/kernel', name + '/bias'))

			if pm.activation == 'relu':
				res = self.ng.relu( (res, name + '/Relu') )
			elif pm.activation == 'softmax':
				res = self.ng.softmax( (res, name + '/Softmax') )
			else:
				assert pm.activation is None
			return res

		return func

	@params([
		('pool_size',(2, 2)), 
		('strides',None), 
		('padding','valid')
	], name='max_pooling2d', skip=['data_format'])
	def MaxPooling2D(self, name, pm):
		def func(input):
			return self.ng.maxpool2d((input, name + '/MaxPool'), pool_size=pm.pool_size,  padding=pm.padding, strides=pm.strides)
		return func
		

	@params([], name='flatten', skip = ['data_format'])
	def Flatten(self, name, pm):
		def func(input):
			return self.ng.reshape((input, name + '/Reshape'))
		return func


	@params([
		('axis', -1), 
		('epsilon',0.001), 

		# ('momentum', 0.99), 
		# ('center', True), 
		# ('scale', True), 
	], name='batch_norm_2d',skip = common_skip([
		'beta', 'gamma', 
		'moving_mean', 'moving_variance'
	]))
	def BatchNormalization(self, name, pm):
		def func(input):
			weights = map(lambda x: name + '/' + x, ('gamma', 'beta', 'moving_mean', 'moving_variance' ))
			return self.ng.batch_norm_2d((input, name + '/FusedBatchNorm'), tuple(weights), 
					eps=pm.epsilon, axis=pm.axis)
		return func

	@params([], name='input', skip = ['shape'])
	def Input(self, name, pm):
		return self.ng.tensor(name)

	@params([
		('activation', NoParam)
	], name='activation')
	def Activation(self, name, pm):
		def func(input):
			return self.ng.activation((input, name + '/' + pm.activation.title()))
		return func

	@params([
		('padding', (1,1))
	], name='zero_padding_2d',skip = ['data_format'])
	def ZeroPadding2D(self, name, pm):
		def func(input):
			return self.ng.zero_padding_2d(
				(input, name + '/Pad'), 
				padding = pm.padding
			)
			
		return func

	@params([], name='avg_pool', skip=['data_format'])
	def GlobalAveragePooling2D(self, name, pm):
		def func(input):
			return self.ng.global_average_pooling_2d((input, name + '/Mean'))
		return func

	@params([], name='Add')
	def Add(self, name, pm):
		def func(inputs):
			return self.ng.add((*inputs, name + '/add'))
		return func







