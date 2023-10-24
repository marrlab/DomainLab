from domainlab.algos.trainers.fbopt_setpoint_ada import is_less_list_all
def test_less_than():
  a = [3, 4, -9, -8]
  b = [1, 0.5, -1, -0.5]
  c = [0.5, 0.25, -0.5, -0.25]
  assert not is_less_list_all(a, b)
  assert is_less_list_all(c, b)
