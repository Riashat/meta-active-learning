from src.utils import RewardProcess


def test_reward_process():
	rp = RewardProcess('marginalacc')
	assert rp.get_reward(0, 1, 4, 4) == 1

	rp = RewardProcess('acc')
	assert rp.get_reward(0, 1, 4, 4) == 1

	rp = RewardProcess('marginallogp')
	assert rp.get_reward(0, 1, 4, 4) == 0

	rp = RewardProcess('logp')
	assert rp.get_reward(0, 1, 4, 4) == 4